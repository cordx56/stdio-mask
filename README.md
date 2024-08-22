# stdio-mask

Mask your private text shown in terminal

## How to use

First, create config file named `.maskrc.toml`

```toml
[check]
text = ["secret"]
[mask]
char = "x"
```

Then, start stdio-mask with arguments that is command you would like to execute

```bash
cargo run -- bash
```

Text "secret" in stdio of bash will be masked by "x"

## How it works

stdio-mask creates pty (pseudo terminal) and output masked pty output to stdout

## 参考にした資料

資料を残してくれた先人たちに感謝します。

- [PTY を使ってシェルの入出力を好きなようにする - Hibariya](https://note.hibariya.org/articles/20150628/pty.html)
- [ターミナルの幅と高さに関するメモ書き](https://zenn.dev/kusaremkn/articles/abdbd2f38c3d98b145eb)
