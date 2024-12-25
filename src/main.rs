use libc::{ioctl, kill, TIOCGWINSZ, TIOCSWINSZ};
use nix::{
    pty::{openpty, Winsize},
    sys::{
        signal::{signal, SigHandler, Signal},
        termios::{
            tcgetattr, tcsetattr, ControlFlags, InputFlags, LocalFlags, OutputFlags, SetArg,
            SpecialCharacterIndices, Termios,
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
use std::os::fd::{AsRawFd, OwnedFd};
use std::sync::{Arc, RwLock};
use std::thread;

use termwiz::{
    caps::Capabilities,
    cell::{Cell, CellAttributes},
    escape::{
        csi::{Edit, EraseInDisplay, CSI},
        parser::Parser,
        Action,
    },
    surface::Surface,
    terminal::{self, buffered::BufferedTerminal, ScreenSize, Terminal, UnixTerminal},
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

pub struct VTerm<R: Read, W: Write> {
    buf: Arc<RwLock<Vec<u8>>>,
    input: R,
    output: W,
}
impl<R, W> VTerm<R, W>
where
    R: Read,
    W: Write,
{
    pub fn new(input: R, output: W) -> Self {
        Self {
            buf: Arc::new(RwLock::new(Vec::new())),
            input,
            output,
        }
    }
    pub fn save_cursor_position(&mut self) {
        self.output.write_all(&[0x1b, b'7']).unwrap();
    }
    pub fn restore_cursor_position(&mut self) {
        self.output.write_all(&[0x1b, b'8']).unwrap();
    }
    pub fn cursor_position(&mut self) -> (usize, usize) {
        let len = { self.buf.read().unwrap().len() };
        self.output.write(&[0x1b]).unwrap();
        self.output.write(b"[6n").unwrap();
        self.flush();
        //std::thread::sleep(std::time::Duration::from_secs(3));
        while 0 != { self.read() } {
            let buf = self.buf.read().unwrap();
            let mut f = File::options().append(true).open("log.txt").unwrap();
            writeln!(f, "buf {}", String::from_utf8_lossy(&buf));
            let buflen = buf.len();
            for i in len..buflen {
                if buf[i] == 0x1b && buf.get(i + 1) == Some(&b'[') {
                    'outer: for j in i + 2..buflen {
                        if buf[j] == b';' {
                            writeln!(f, "i1 {i}:{j}, {:?}", &buf[i + 2..j]);
                            let row = String::from_utf8_lossy(&buf[i + 2..j]).parse().unwrap();
                            writeln!(f, "{row}");
                            for k in j + 1..buflen {
                                writeln!(f, "i2 {j}:{k}, {:?}", &buf[j + 1..k + 1]);
                                if buf[k] == b'R' {
                                    let col =
                                        String::from_utf8_lossy(&buf[j + 1..k]).parse().unwrap();
                                    writeln!(f, "OK {i}:{k}");
                                    /*
                                    let tail = buf[k + 1..].to_vec();
                                    let head = buf[..i].to_vec();
                                    drop(buf);
                                    let mut buf = self.buf.write().unwrap();
                                    *buf = head;
                                    buf.extend_from_slice(&tail);
                                    */
                                    return (row, col);
                                } else if !buf[k].is_ascii_digit() {
                                    writeln!(f, "i2break {j}:{k}, {:?}", &buf[j + 1..k]);
                                    break 'outer;
                                }
                            }
                        } else if !buf[j].is_ascii_digit() {
                            break;
                        }
                    }
                }
            }
        }
        (0, 0)
    }
    pub fn read(&mut self) -> usize {
        let mut buf = [0; 4096];
        if let Ok(len) = self.input.read(&mut buf) {
            self.buf.write().unwrap().extend_from_slice(&buf[..len]);
            return len;
        }
        return 0;
    }
    pub fn write(&mut self, buf: &[u8]) {
        self.output.write(buf).unwrap();
    }
    pub fn render(&mut self) {
        self.clear();
        self.flush();
        self.output.write(&self.buf.read().unwrap()).unwrap();
        self.flush();
    }
    pub fn flush(&mut self) {
        self.output.flush().unwrap();
    }
    pub fn move_to(&mut self, row: usize, col: usize) {
        self.output.write(&[0x1b, b'[']).unwrap();
        self.output
            .write(format!("{};{}H", row, col).as_bytes())
            .unwrap();
    }
    pub fn clear(&mut self) {
        self.output.write(&[0x1b, b'[', b'J', b'2']).unwrap();
        self.move_to(0, 0);
    }
}
impl Clone for VTerm<File, File> {
    fn clone(&self) -> Self {
        Self {
            buf: self.buf.clone(),
            input: self.input.try_clone().unwrap(),
            output: self.output.try_clone().unwrap(),
        }
    }
}

pub struct VirtualScreen {
    term: BufferedTerminal<UnixTerminal>,
    master: File,
    //surf: Surface,
    remain: Vec<u8>,
    //actions: Vec<Action>,
    screen: Vec<Vec<Option<(usize, char)>>>,
    //chars: Vec<Vec<Option<(usize, char)>>>,
}
impl VirtualScreen {
    pub fn new(col: usize, row: usize) -> VirtualScreen {
        let remain = Vec::new();
        //let actions = Vec::new();
        //let (master, slave) = newpty();
        //let mut term = UnixTerminal::new(Capabilities::new_from_env().unwrap()).unwrap();
        //let screen_size = term.get_screen_size().unwrap();
        //let (master, slave) = newpty();
        let (master, slave) = newpty();
        let mut term =
            UnixTerminal::new_with(Capabilities::new_from_env().unwrap(), &slave, &slave).unwrap();
        //let surf = Surface::new(screen_size.cols, screen_size.rows);
        //term.set_screen_size(screen_size).unwrap();
        let mut term = BufferedTerminal::new(term).unwrap();
        term.resize(col, row);
        //term.terminal().set_raw_mode().unwrap();
        let (w, h) = term.dimensions();
        let screen = vec![vec![None; w + 1]; h + 1];
        VirtualScreen {
            remain,
            //actions,
            term,
            master: File::from(master),
            //surf,
            screen,
        }
    }
    fn flush(&mut self) {
        self.master.flush().unwrap();
        self.term.flush().unwrap();
    }
    fn write(&mut self, s: &str) {
        self.master.write_all(s.as_bytes()).unwrap();
        self.term.add_change(s);
    }
    fn input<'a>(&'a mut self, bytes: &[u8]) -> String {
        // -> &'a Surface {
        // -> &'a BufferedTerminal<UnixTerminal> {
        //let mut term = UnixTerminal::new(Capabilities::new_from_env().unwrap()).unwrap();

        self.remain.extend_from_slice(bytes);

        let mut parser = Parser::new();
        let mut actions = Vec::new();
        let mut remi = 0;
        loop {
            if let Some((act, i)) = parser.parse_first(&self.remain[remi..]) {
                remi += i;
                match act {
                    Action::PrintString(s) => {
                        let chars: Vec<_> =
                            s.chars().into_iter().map(|c| Action::Print(c)).collect();
                        actions.extend_from_slice(&chars);
                    }
                    _ => {
                        actions.push(act);
                    }
                }
            } else {
                break;
            }
        }
        self.remain = self.remain[remi..].to_vec();

        /*
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
        */

        //self.term.terminal().enter_alternate_screen().unwrap();
        //self.term.add_change(
        //self.term.terminal().enter_alternate_screen().unwrap();
        /*
        self.term.add_change(format!(
            "{}",
            CSI::Edit(Edit::EraseInDisplay(EraseInDisplay::EraseDisplay))
        ));
        */
        //self.surf.flush().unwrap();
        //self.term.flush().unwrap();
        for i in 0..actions.len() {
            let action = &actions[i];
            match action {
                Action::Print(c) => {
                    self.flush();
                    //self.master.flush().unwrap();
                    //self.term.flush().unwrap();
                    let (x, y) = self.term.cursor_position();
                    //println!("({},{})", x, y);
                    //let (x, y) = self.surf.cursor_position();
                    self.screen[y][x] = Some((i, *c));
                }
                _ => {}
            }
            self.write(&format!("{}", action));
            /*self.master
                .write_all(format!("{}", action).as_bytes())
                .unwrap();
            */
            self.flush();
            //self.master.flush().unwrap();
            //self.term.flush().unwrap();

            let search = "ma21029";
            for rowi in 0..self.screen.len() {
                'outer: for coli in 0..self.screen[rowi].len() {
                    let mut indices = Vec::new();
                    for i in 0..search.chars().count() {
                        if let Some(Some((i, c))) = self.screen[rowi].get(coli + i) {
                            if Some(*c) != search.chars().nth(*i) {
                                continue 'outer;
                            }
                            indices.push(*i);
                        }
                    }
                    for i in indices {
                        actions[i] = Action::Print('x');
                    }
                }
            }
        }
        //self.master.flush().unwrap();
        //self.term.flush().unwrap();
        //let mut actions = self.actions.clone();

        //self.term.flush().unwrap();

        /*
        let mut f = File::options()
            .create_new(true)
            .append(true)
            .open("log.txt")
            .unwrap();
        */
        /*
        self.write(&format!(
            "{}",
            CSI::Edit(Edit::EraseInDisplay(EraseInDisplay::EraseDisplay))
        ));
        */
        for action in &actions {
            self.write(&format!("{action}"));
        }
        //self.master.flush().unwrap();
        self.flush();
        //self.term.terminal().exit_alternate_screen().unwrap();
        //self.term.flush().unwrap();
        //self.term.terminal().exit_alternate_screen().unwrap();
        actions
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
fn set_raw() -> impl Fn() {
    let stdin = io::stdin();
    // save current tty attr for restoring
    let mut attr = getattr();
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
    tcsetattr(&stdin, SetArg::TCSAFLUSH, &attr).unwrap();
    return move || {
        tcsetattr(&stdin, SetArg::TCSAFLUSH, &current_attr).unwrap();
    };
}
fn getattr() -> Termios {
    let stdin = io::stdin();
    tcgetattr(&stdin).unwrap()
}

fn newpty() -> (OwnedFd, OwnedFd) {
    let winsize = get_winsize().unwrap();
    let pty = openpty(&winsize, &getattr()).unwrap();
    let master = pty.master;
    let slave = pty.slave;
    (master, slave)
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

    let (master, slave) = newpty();
    match unsafe { fork().unwrap() } {
        ForkResult::Parent { child } => {
            // change signal actions
            // set global child pid to hook signal
            unsafe {
                CHILD.write(child);
                PTY_FD.write(slave.as_raw_fd());
                signal(Signal::SIGWINCH, SigHandler::Handler(resize_window)).unwrap();
            }

            let reset = set_raw();

            //let (tx, rx) = mpsc::channel();

            //let size = get_winsize().unwrap();
            //let mut vt = VirtualScreen::new(size.ws_col as usize, size.ws_row as usize);

            //let term = UnixTerminal::new(Capabilities::new_from_env().unwrap()).unwrap();
            //let mut term = BufferedTerminal::new(term).unwrap();
            //term.terminal().set_raw_mode().unwrap();

            // open pty master as File
            //let master_file = File::from(master);
            //let mut master = master_file.try_clone().unwrap();

            /*
            let mut master_term = BufferedTerminal::new(
                UnixTerminal::new(Capabilities::new_from_env().unwrap()).unwrap(),
            )
            .unwrap();
            master_term.terminal().set_raw_mode().unwrap();
            */

            // read child's output and output to stdout
            //let mut master_file = master.try_clone().unwrap();
            //let mut log = fs::File::create_new("log.txt").unwrap();
            let (master2, slave2) = newpty();
            match unsafe { fork().unwrap() } {
                ForkResult::Parent { child: child2 } => {
                    unsafe {
                        CHILD.write(child2);
                        PTY_FD.write(slave2.as_raw_fd());
                        signal(Signal::SIGWINCH, SigHandler::Handler(resize_window)).unwrap();
                    }

                    let reset = set_raw();

                    let master2_file = File::from(master2);
                    let mut master2 = master2_file.try_clone().unwrap();
                    thread::spawn(move || {
                        let stdin = io::stdin();
                        for byte in stdin.bytes() {
                            let byte = byte.unwrap();
                            //tx.send(byte).unwrap();
                            master2.write(&[byte]).unwrap();
                            //master2.flush().unwrap();
                        }
                    });

                    thread::spawn(move || {
                        let mut master2 = master2_file;
                        let mut stdout = io::stdout();
                        let mut buf = vec![0; 4096];
                        while let Ok(len) = master2.read(&mut buf) {
                            if len == 0 {
                                break;
                            }
                            stdout.write_all(&buf[..len]).unwrap();
                            stdout.flush().unwrap();
                        }
                    });

                    loop {
                        let status = waitpid(child, None).unwrap();
                        match status {
                            WaitStatus::Exited(_, _) => break,
                            _ => {}
                        }
                    }

                    // restore terminal settings
                    reset();
                }
                ForkResult::Child => {
                    unsafe {
                        CHILD.write(child);
                        PTY_FD.write(slave.as_raw_fd());
                        signal(Signal::SIGWINCH, SigHandler::Handler(resize_window)).unwrap();
                    }

                    let reset = set_raw();

                    let slave2_file = File::from(slave2);
                    let master_file = File::from(master);
                    //let mut master = master_file;//.try_clone().unwrap();

                    //dup2(slave2.as_raw_fd(), master.as_raw_fd()).unwrap();
                    /*
                    let term =
                        UnixTerminal::new_with(Capabilities::new_from_env().unwrap(), &master, &gg)
                            .unwrap();
                    let mut term = BufferedTerminal::new(term).unwrap();
                    */
                    let master = master_file.try_clone().unwrap();
                    let mut term =
                        VTerm::new(master.try_clone().unwrap(), master.try_clone().unwrap());

                    //thread::spawn(move || {
                    let mut buf = [0; 4096];
                    let mut parser = Parser::new();
                    let mut rem = 0;
                    //let mut master = master_file;
                    //let mut slave2 = slave2_file;
                    //let mut out = 0;

                    let mut slave2 = slave2_file.try_clone().unwrap();
                    let t = term.clone();
                    thread::spawn(move || {
                        let mut term = t;
                        let mut out = 0;
                        loop {
                            let o = term.read();
                            //println!("{o}");
                            if o == 0 {
                                break;
                            }
                            slave2
                                .write(&term.buf.read().unwrap()[out..out + o])
                                .unwrap();
                            //slave2.flush().unwrap();
                            out += o;
                        }
                    });

                    let mut slave2 = slave2_file.try_clone().unwrap();
                    while let Ok(len) = slave2.read(&mut buf[rem..]) {
                        //jhprintln!("{}", len);
                        if len == 0 {
                            break;
                        }
                        let mut actions = Vec::new();
                        let mut read_len = rem;
                        while let Some((act, i)) = parser.parse_first(&buf[read_len..rem + len]) {
                            read_len += i;
                            match act {
                                Action::PrintString(s) => {
                                    for ch in s.chars() {
                                        actions.push(Action::Print(ch));
                                    }
                                }
                                _ => {
                                    actions.push(act);
                                }
                            }
                        }
                        for i in read_len..rem + len {
                            buf[i - read_len] = buf[i];
                        }
                        rem = rem + len - read_len;
                        let mut screen = vec![vec![' '; 4096]; 4096];
                        for act in &actions {
                            let (row, col) = term.cursor_position();
                            match act {
                                Action::Print(ch) => {
                                    screen[row][col] = *ch;
                                }
                                _ => {}
                            }
                            term.write(format!("{act}").as_bytes());
                            term.flush();
                        }
                        /*
                        for rowi in 0..screen.len() {
                            for coli in 0..screen[rowi].len() {
                                if screen[rowi][coli] == 'm' {
                                    let (row, col) = term.cursor_position();
                                    term.move_to(rowi, coli);
                                    term.write("x".as_bytes());
                                    term.move_to(row, col);
                                }
                            }
                        }
                        */
                        term.flush();
                    }
                    //});
                    reset();

                    loop {}
                    /*
                    let mut buf = [0; 4096];
                    while let Ok(len) = master.read(&mut buf) {
                        if len == 0 {
                            break;
                        }
                        slave2.write_all(&buf[..len]).unwrap();
                        slave2.flush().unwrap();
                    }
                    */
                }
            }

            reset();

            /*
                thread::spawn(move || {
                    let mut stdout = io::stdout();

                    let mut master2 = File::from(master2);
                    //dup2(slave2.as_raw_fd(), master_file.as_raw_fd()).unwrap();
                    //let mut slave2 = File::from(slave2);
                    //let mut buf = CheckBuf::new(check_list, mask);
                    //let mut buffer = Vec::new();
                    //let mut display =
                    //terminal::new_terminal(Capabilities::new_from_env().unwrap()).unwrap();
                    //let size = display.get_screen_size().unwrap();
                    //let mut surf = surface::Surface::new(size.cols, size.rows);
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

                    //let mut virt = VirtualScreen::new();

                    //let mut term =
                    //    terminal::new_terminal(Capabilities::new_from_env().unwrap()).unwrap();
                    /*
                    let mut bufterm = UnixTerminal::new_with(
                        Capabilities::new_from_env().unwrap(),
                        &file,
                        &file,
                    )
                    .unwrap();
                    bufterm.set_raw_mode().unwrap();
                    let mut bufterm = BufferedTerminal::new(bufterm).unwrap();
                    */

                    //let screen_size = term.terminal().get_screen_size().unwrap();

                    //let (w, h) = term.dimensions();
                    //let mut bufterm =
                    //    terminal::new_terminal(Capabilities::new_from_env().unwrap()).unwrap();
                    //let mut bufterm = terminal::buffered::BufferedTerminal::new(bufterm).unwrap();
                    //let mut virt = VirtualScreen::new(screen_size);

                    /*
                    let mut seq = 0;
                    loop {
                        if bufterm.has_changes(seq) {
                            let (newseq, changes) = bufterm.get_changes(seq);
                            seq = newseq;
                            term.add_changes(changes.to_vec());
                            //term.draw_from_screen(&bufterm, 0, 0);
                            term.flush().unwrap();
                        }
                    }
                    */
                    let mut buf = vec![0; 4096];
                    while let Ok(len) = master.read(&mut buf) {
                        if len == 0 {
                            break;
                        }
                        //master2.write_all(&buf[..len]).unwrap();
                        //master2.flush().unwrap();
                        stdout.write_all(&buf[..len]).unwrap();
                        stdout.flush().unwrap();
                    }
                    /*
                    while let Ok(byte) = rx.recv() {
                        //slave2.write(&[byte]).unwrap();
                        //slave2.write_all(&buf[..len]).unwrap();
                        //slave2.flush().unwrap();

                        master_file.write_all(&[byte]).unwrap();
                        master_file.flush().unwrap();
                        /*
                        while let Ok(len) = master2.read(&mut buf) {
                            if len == 0 {
                                break;
                            }
                            stdout.write_all(&buf[..len]).unwrap();
                        }
                        */
                        stdout.write_all(&[byte]).unwrap();
                        stdout.flush().unwrap();

                        //let s = vt.input(&buf[..len]);
                        //term.draw_from_screen(&s, 0, 0);
                        //term.add_change(String::from_utf8_lossy(&buf[0..len]));
                        //term.add_change(s);
                        //term.flush().unwrap();
                    }
                    */

                    /*k
                    let mut rest = Vec::new();
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
                        //let mut all_buffer = String::new();
                        let mut buf = vec![0; 4096];
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

                            //let actions = virt.input(&buf[0..len]);

                            //let parsed = parser.parse_as_vec(&buf[0..len]);

                            //let mut bufterm =
                            //    terminal::new_terminal(Capabilities::new_from_env().unwrap()).unwrap();
                            //bufterm.set_raw_mode().unwrap();
                            //let mut bufterm = BufferedTerminal::new(bufterm).unwrap();

                            let mut parser = Parser::new();
                            let mut i = 0;
                            let mut actions = Vec::new();
                            rest.extend_from_slice(&buf[0..len]);
                            loop {
                                if let Some((act, read)) = parser.parse_first(&rest[i..len]) {
                                    i += read;
                                    //actions.push(act);
                                    act.append_to(&mut actions);
                                    //bufterm.add_change(act.);
                                } else {
                                    rest = rest[i..len].to_vec();
                                    //rest.extend_from_slice(&buf[i..len]);
                                    break;
                                }
                            }
                            //bufterm.add_change(actions.iter().map(|v| format!("{v}")).collect::<Vec<_>>().join(""));
                            //bufterm.flush().unwrap();
                            //term.add_change(&actions);

                            //surf.add_change(&parsed.into_iter().map(|v| format!("{v}")).collect::<Vec<_>>().join(""));
                            /*
                            for p in parsed {
                                term.add_change(format!("{}", p));
                                //stdout.write(p.to_string().as_bytes());
                            }
                            stdout.flush().unwrap();
                            */
                            term.add_change(
                                actions
                                    .iter()
                                    .map(|v| format!("{v}"))
                                    .collect::<Vec<_>>()
                                    .join(""),
                            );
                            term.flush().unwrap();
                            //let changes = term.diff_screens(&surf);
                            //term.add_changes(changes);
                            //term.add_change(&all_buffer);
                            //term.add_change(parsed.into_iter().map(|v| format!("{v}")).collect::<Vec<_>>().join(""));
                            //term.draw_from_screen(&bufterm, 0, 0);
                            //term.flush().unwrap();
                        } else {
                            break;
                        }
                        //term.flush().unwrap();
                    }
                    */

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
            */

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
            let mut master = master_file.try_clone().unwrap();
            //let mut slave2 = slave2_file.try_clone().unwrap();
            thread::spawn(move || {
                let mut stdin = io::stdin();
                for byte in stdin.bytes() {
                    let byte = byte.unwrap();
                    tx.send(byte).unwrap();
                    master.write(&[byte]).unwrap();
                    master.flush().unwrap();
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
            */

            // wait child process exit
            /*
            loop {
                let status = waitpid(child, None).unwrap();
                match status {
                    WaitStatus::Exited(_, _) => break,
                    _ => {}
                }
            }

            // restore terminal settings
            //term.terminal().set_cooked_mode().unwrap();
            tcsetattr(&stdin, SetArg::TCSAFLUSH, &current_attr).unwrap();
            */
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
