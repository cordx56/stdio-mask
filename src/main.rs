use libc::{ioctl, kill, TIOCGWINSZ, TIOCSWINSZ};
use nix::{
    pty::{openpty, Winsize},
    sys::{
        signal::{signal, SigHandler, Signal},
        termios::{
            tcgetattr, tcsetattr, ControlFlags, InputFlags, LocalFlags, OutputFlags, SetArg,
            SpecialCharacterIndices,
        },
        wait::{waitpid, WaitStatus},
    },
    unistd::{dup2, execvp, fork, ForkResult, Pid},
};
use serde::Deserialize;
use std::collections::VecDeque;
use std::env;
use std::ffi::c_int;
use std::ffi::CString;
use std::fs;
use std::fs::File;
use std::io::{self, Read, Write};
use std::mem::MaybeUninit;
use std::os::fd::AsRawFd;
use std::sync::mpsc;
use std::thread;

use termwiz::{
    caps::Capabilities,
    cell::{Cell, CellAttributes},
    escape::{parser::Parser, Action},
    surface,
    terminal::{self, buffered::BufferedTerminal, Terminal, UnixTerminal},
};

#[derive(Deserialize)]
struct Check {
    text: Vec<String>,
}
#[derive(Deserialize)]
struct Mask {
    char: String,
}
#[derive(Deserialize)]
struct Config {
    check: Check,
    mask: Mask,
}

static mut CHILD: MaybeUninit<Pid> = MaybeUninit::uninit();
extern "C" fn send_signal(sig: c_int) {
    unsafe {
        kill(CHILD.assume_init().as_raw(), sig);
    }
}

/// Output checking buffer
struct CheckBuf {
    check: Vec<String>,
    buf: VecDeque<u8>,
    mask: String,
}
impl CheckBuf {
    pub fn new<I>(ss: I, mask: String) -> Self
    where
        I: IntoIterator,
        I::Item: ToString,
    {
        Self {
            check: ss.into_iter().map(|v| v.to_string()).collect(),
            buf: VecDeque::with_capacity(32),
            mask,
        }
    }
    fn push(&mut self, byte: u8) -> Option<u8> {
        self.buf.push_back(byte);
        if self.buf.len() < self.buf.capacity() {
            None
        } else {
            self.buf.pop_front()
        }
    }
    fn check(&self) -> Option<usize> {
        'check: for check in &self.check {
            let check_bytes = check.as_bytes();
            let check_len = check_bytes.len();
            let buf_len = self.buf.len();
            if check_len <= buf_len {
                let vec_range = (buf_len - check_len)..buf_len;
                for (vi, ci) in vec_range.clone().zip(0..) {
                    if self.buf[vi] != check_bytes[ci] {
                        continue 'check;
                    }
                }
                return Some(check_len);
            }
        }
        None
    }
    fn back(&self) -> Vec<u8> {
        let mut vec = vec![];
        if let Some(len) = self.check() {
            vec.push(0x1b);
            vec.extend_from_slice(format!("[{}D{}", len, self.mask.repeat(len)).as_bytes());
            vec
        } else {
            vec
        }
    }
    fn output(&mut self, byte: u8) -> Vec<u8> {
        self.push(byte);
        let mut ret = vec![byte];
        ret.extend_from_slice(&self.back());
        ret
    }
}

/// Get current window size
fn get_winsize() -> Result<Winsize, i32> {
    let stdin = io::stdin().as_raw_fd();
    let winsize: MaybeUninit<Winsize> = MaybeUninit::uninit();
    let res = unsafe {
        let res = ioctl(stdin, TIOCGWINSZ, winsize.as_ptr());
        if res == 0 {
            Ok(winsize.assume_init())
        } else {
            Err(res)
        }
    };
    res
}
static mut PTY_FD: MaybeUninit<i32> = MaybeUninit::uninit();
extern "C" fn resize_window(_: c_int) {
    let winsize = get_winsize().unwrap();
    unsafe {
        let res = ioctl(PTY_FD.assume_init(), TIOCSWINSZ, &winsize);
        if res != 0 {
            Err::<(), _>(res).unwrap();
        }
    }
}

fn clear(mut w: impl Write) {
    w.write(&[0x1b]).unwrap();
    w.write("[1;1H".as_bytes()).unwrap();
    w.write(&[0x1b]).unwrap();
    w.write("[0J".as_bytes()).unwrap();
}
fn save_cursor_pos(mut w: impl Write) {
    w.write(&[0x1b, '7' as u8]).unwrap();
}
fn restore_cursor_pos(mut w: impl Write) {
    w.write(&[0x1b, '8' as u8]).unwrap();
}
fn mv_cursor(mut w: impl Write, row: usize, col: usize) {
    w.write(&[0x1b]).unwrap();
    w.write(format!("[{};{}H", row, col).as_bytes()).unwrap();
}
fn replace(mut w: impl Write, left: i32, ch: char) {
    w.write(&[0x1b]).unwrap();
    w.write(format!("[{}D", left).as_bytes()).unwrap();
    for _ in 0..left {
        w.write(&[ch as u8]).unwrap();
    }
}
fn alter(mut w: impl Write) {
    w.write(&[0x1b]).unwrap();
    w.write(b"[?1049h").unwrap();
}
fn noalter(mut w: impl Write) {
    w.write(&[0x1b]).unwrap();
    w.write(b"[?1049l").unwrap();
}
fn get_cursor_pos(mut r: impl Read, mut w: impl Write) -> (usize, usize, Vec<u8>) {
    let mut row = 0;
    w.write(&[0x1b]).unwrap();
    w.write("[6n".as_bytes()).unwrap();
    w.flush().unwrap();
    let mut buf = Vec::new();
    let mut return_buf = Vec::new();
    r.read_exact(&mut [0]).unwrap();
    loop {
        let mut b = [0; 1];
        r.read_exact(&mut b).unwrap();
        let c = b[0];
        if c == ';' as u8 {
            for i in (buf.len() - 1..=0).rev() {
                if buf[i] == '[' as u8 {
                    return_buf = buf[0..i - 1].to_vec();
                    row = String::from_utf8_lossy(&buf[i + 1..]).parse().unwrap();
                }
            }
            buf = Vec::new();
        } else if c == 'R' as u8 {
            let col = String::from_utf8_lossy(&buf).parse().unwrap();
            return (row, col, return_buf);
        } else {
            buf.push(c);
        }
    }
}

pub struct VirtualScreen {
    //term: BufferedTerminal<T>,
    actions: Vec<Action>,
    //chars: Vec<Vec<Option<(usize, char)>>>,
}
impl VirtualScreen {
    pub fn new() -> VirtualScreen {
        let actions = Vec::new();
        VirtualScreen { actions }
    }
    fn input(&mut self, bytes: &[u8]) -> String {
        let mut term = UnixTerminal::new(Capabilities::new_from_env().unwrap()).unwrap();
        term.set_raw_mode().unwrap();
        let mut term = BufferedTerminal::new(term).unwrap();
        let (w, h) = term.dimensions();
        let mut chars = vec![vec![None; w + 1]; h + 1];

        let mut parser = Parser::new();
        let mut actions = Vec::new();
        for action in parser.parse_as_vec(bytes) {
            match action {
                Action::PrintString(s) => {
                    let chars: Vec<_> = s.chars().into_iter().map(|c| Action::Print(c)).collect();
                    actions.extend_from_slice(&chars);
                }
                _ => {
                    actions.push(action);
                }
            }
        }

        self.actions.extend_from_slice(&actions);
        for (i, action) in self.actions.iter().enumerate() {
            match action {
                Action::Print(c) => {
                    let (x, y) = term.cursor_position();
                    chars[y][x] = Some((i, *c));
                }
                _ => {}
            }
            term.add_change(format!("{}", action));
            term.flush().unwrap();
        }
        for rowi in 0..chars.len() {
            let mut indices = Vec::new();
            let mut s = String::new();
            for coli in 0..chars[rowi].len() {
                if let Some((i, c)) = &chars[rowi][coli] {
                    indices.push(*i);
                    s.push(*c);
                    if s == "ma21029" {
                        for i in &indices {
                            self.actions[*i] = Action::Print('x');
                        }
                        s = String::new();
                    }
                } else {
                    indices = Vec::new();
                    s = String::new();
                }
            }
        }
        self.actions
            .iter()
            .map(|v| format!("{}", v))
            .collect::<Vec<_>>()
            .join("")
    }
}

fn check_actions(actions: &[Action], check: &str) -> Vec<Action> {
    let mut res = Vec::new();
    for i in 0..actions.len() {
        match &actions[i] {
            Action::PrintString(s) => {
                for c in s.chars() {
                    res.push(Action::Print(c));
                }
            }
            _ => {
                res.push(actions[i].clone());
            }
        }
    }
    let mut checked = Vec::new();
    for i in 0..res.len() {
        match &res[i] {
            Action::Print(c) => {
                if let Some(checkch) = check.chars().nth(checked.len()) {
                    if *c == checkch {
                        checked.push(i);
                    } else {
                        checked = Vec::new();
                    }
                } else {
                    for i in &checked {
                        res[*i] = Action::Print('x');
                    }
                }
            }
            _ => {}
        }
    }
    res
}

fn main() {
    /*
    let config = fs::read_to_string(".maskrc.toml").unwrap();
    let config: Config = toml::from_str(&config).unwrap();
    let check_list = config.check.text.clone();
    let mask = config.mask.char.clone();
    */

    let mut a = env::args();
    a.next();
    let program = match a.next() {
        Some(v) => v,
        None => return,
    };
    let program = CString::new(program).unwrap();
    let mut args = vec![program.clone()];
    args.extend(
        a.map(|v| CString::new(v))
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
    );

    let stdin = io::stdin();
    let mut attr = tcgetattr(&stdin).unwrap();
    let winsize = get_winsize().unwrap();
    let pty = openpty(&winsize, &attr).unwrap();
    let master = pty.master;
    let slave = pty.slave;
    match unsafe { fork().unwrap() } {
        ForkResult::Parent { child } => {
            // save current tty attr for restoring
            let current_attr = attr.clone();

            // set terminal settings
            attr.local_flags &=
                !(LocalFlags::ECHO | LocalFlags::ICANON | LocalFlags::IEXTEN | LocalFlags::ISIG);
            attr.input_flags &= !(InputFlags::BRKINT
                | InputFlags::ICRNL
                | InputFlags::INPCK
                | InputFlags::ISTRIP
                | InputFlags::IXON);
            attr.control_flags &= !(ControlFlags::CSIZE | ControlFlags::PARENB);
            attr.control_flags |= ControlFlags::CS8;
            attr.output_flags &= !(OutputFlags::OPOST);
            attr.control_chars[SpecialCharacterIndices::VMIN as usize] = 1;
            attr.control_chars[SpecialCharacterIndices::VTIME as usize] = 0;
            //tcsetattr(&stdin, SetArg::TCSAFLUSH, &attr).unwrap();

            // change signal actions
            // set global child pid to hook signal
            unsafe {
                CHILD.write(child);
                PTY_FD.write(slave.as_raw_fd());
                signal(Signal::SIGWINCH, SigHandler::Handler(resize_window)).unwrap();
            }

            //let (tx, rx) = mpsc::channel();

            // open pty master as File
            let master = File::from(master);

            // read child's output and output to stdout
            let mut file = master.try_clone().unwrap();
            //let mut log = fs::File::create_new("log.txt").unwrap();
            thread::spawn(move || {
                let mut stdout = io::stdout();
                //let mut buf = CheckBuf::new(check_list, mask);
                //let mut buffer = Vec::new();
                let mut display =
                    terminal::new_terminal(Capabilities::new_from_env().unwrap()).unwrap();
                let size = display.get_screen_size().unwrap();
                let mut surf = surface::Surface::new(size.cols, size.rows);
                /*
                let mut bufterm = terminal::UnixTerminal::new_with(
                    Capabilities::new_from_env().unwrap(),
                    &io::stdin().as_raw_fd(),
                    &file.as_raw_fd(),
                )
                .unwrap();
                bufterm.set_raw_mode().unwrap();
                //terminal::new_terminal(Capabilities::new_from_env().unwrap()).unwrap()
                let mut bufterm = terminal::buffered::BufferedTerminal::new(bufterm).unwrap();
                */
                /*
                let mut term = terminal::UnixTerminal::new_with(
                    Capabilities::new_from_env().unwrap(),
                    &file.as_raw_fd(),
                    &io::stdout(),
                )
                .unwrap();
                */
                let mut term =
                    terminal::new_terminal(Capabilities::new_from_env().unwrap()).unwrap();
                term.set_raw_mode().unwrap();
                let mut term = terminal::buffered::BufferedTerminal::new(term).unwrap();
                let (w, h) = term.dimensions();
                let mut bufterm =
                    terminal::new_terminal(Capabilities::new_from_env().unwrap()).unwrap();
                let mut bufterm = terminal::buffered::BufferedTerminal::new(bufterm).unwrap();
                let mut parser = Parser::new();
                loop {
                    //bufterm.flush().unwrap();
                    /*k
                    for cells in bufterm.screen_cells().into_iter() {
                        for cell in cells {
                            //println!("{cell:?}");
                            if let Some(pos) = cell.str().find("y") {
                                let new = Cell::new('x', cell.attrs().clone());
                                *cell = new;
                            }
                        }
                    }
                    */
                    //let seq = term.draw_from_screen(&bufterm, 0, 0);
                    let mut all_buffer = String::new();
                    let mut buf = vec![0; 4096];
                    let mut virt = VirtualScreen::new();
                    if let Ok(len) = file.read(&mut buf) {
                        if len == 0 {
                            break;
                        }
                        //let actions = check_actions(&parser.parse_as_vec(&buf[0..len]), "ma21029");

                        /*
                        let actions = &parser.parse_as_vec(&buf[0..len]);
                        let changes = actions
                            .into_iter()
                            .map(|v| format!("{}", v))
                            .collect::<Vec<String>>()
                            .join("");
                        //all_buffer.push_str(&String::from_utf8_lossy(&buf[0..len]));
                        all_buffer.push_str(&changes);
                        bufterm.add_change(&changes);
                        let cells = bufterm.screen_cells();
                        for line in cells {
                            for cell in line {
                                let new_str = parser.parse_as_vec(
                                    cell.str().replace("ma21029", "xxxxxxx").as_bytes(),
                                );
                                *cell = Cell::new_grapheme(&new_str, cell.attrs().clone(), None);
                            }
                        }
                        */

                        let actions = virt.input(&buf[0..len]);

                        let bufterm =
                            terminal::new_terminal(Capabilities::new_from_env().unwrap()).unwrap();
                        let mut bufterm = BufferedTerminal::new(bufterm).unwrap();
                        bufterm.add_change(&actions);

                        //bufsurf.add_change(&all_buffer);
                        //let changes = term.diff_screens(&bufsurf);
                        //term.add_change(&all_buffer);
                        term.draw_from_screen(&bufterm, 0, 0);
                        term.flush().unwrap();
                    } else {
                        break;
                    }
                    //term.flush().unwrap();
                }
                /*
                for byte in file.bytes() {
                    let byte = byte.unwrap();
                    bufterm.add_change(byte as char);
                    display.render(&bufterm.get_changes(0).1);
                }
                */

                /*
                clear(&stdout);
                stdout.flush().unwrap();
                let rewrite = "yuk".as_bytes();
                let mut byte = [0];
                while file.read_exact(&mut byte).is_ok() {
                    let byte = byte[0];
                    buffer.push(byte);

                    alter(&stdout);
                    mv_cursor(&stdout, 1, 1);
                    stdout.flush().unwrap();
                    let mut screen = vec![vec![0; 2048]; 2048];
                    let mut rest = Vec::new();
                    for i in 0..buffer.len() {
                        let b = buffer[i];
                        write!(log, "{:?}", String::from_utf8_lossy(&buffer));
                        let (row, col, buf) = get_cursor_pos(&file, &stdout);
                        //println!("{};{}", row, col);
                        rest = buf;
                        write!(
                            log,
                            "{}:{}:{:?}",
                            row,
                            col,
                            String::from_utf8_lossy(&buffer)
                        );
                        screen[row][col] = b;
                        stdout.write_all(&[b]).unwrap();
                        stdout.flush().unwrap();
                    }
                    buffer.extend_from_slice(&rest);
                    noalter(&stdout);
                    stdout.flush().unwrap();

                    // save screen
                    //stdout.write_all(&[0x1b]).unwrap();
                    //stdout.write_all("[?47h".as_bytes()).unwrap();

                    //stdout.write_all(&buffer).unwrap();

                    // restore screen
                    //stdout.write_all(&[0x1b]).unwrap();
                    //stdout.write_all("[?47l".as_bytes()).unwrap();

                    //stdout.flush().unwrap();
                    /*
                    let out = buf.output(byte);
                    stdout.write_all(&out).unwrap();
                    stdout.flush().unwrap();
                    */
                }
                */
            });

            // read from stdin and send to child's stdin
            /*
            let mut file = master.try_clone().unwrap();
            thread::spawn(move || {
                let stdin = io::stdin();
                for byte in stdin.bytes() {
                    let byte = byte.unwrap();
                    file.write_all(&[byte]).unwrap();
                    file.flush().unwrap();
                }
            });
            */
            let mut file = master.try_clone().unwrap();
            thread::spawn(move || {
                let mut stdin = io::stdin();
                for byte in stdin.bytes() {
                    let byte = byte.unwrap();
                    file.write(&[byte]).unwrap();
                    file.flush().unwrap();
                }
                /*
                loop {
                    if let Ok(len) = stdin.read(&mut buf) {
                        if len == 0 {
                            break;
                        }
                        let actions = parser.parse_as_vec(&buf[0..len]);
                        tx.send(
                            actions
                                .into_iter()
                                .map(|v| format!("{}", v))
                                .collect::<Vec<String>>()
                                .join("\n"),
                        )
                        .unwrap();
                    } else {
                        break;
                    }
                }
                */
            });

            // wait child process exit
            loop {
                let status = waitpid(child, None).unwrap();
                match status {
                    WaitStatus::Exited(_, _) => break,
                    _ => {}
                }
            }

            // restore terminal settings
            tcsetattr(&stdin, SetArg::TCSAFLUSH, &current_attr).unwrap();
        }
        ForkResult::Child => {
            // duplicate pty slave FD to stdio
            let slave = slave.as_raw_fd();
            dup2(slave, io::stdin().as_raw_fd()).unwrap();
            dup2(slave, io::stdout().as_raw_fd()).unwrap();
            dup2(slave, io::stderr().as_raw_fd()).unwrap();

            // start command
            execvp(&program, &args).unwrap();
        }
    }
}
