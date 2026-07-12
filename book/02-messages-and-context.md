# Chapter 2: Messages and Context

## Multimodal messages

Beyond plain text, `Message` supports images, audio, and video.
Use the factory classmethods to build rich content from local files or URLs:

```python {name=ch02_multimodal_examples}
# Local image (base64-encoded automatically)
# image_msg = Message.local_image("/path/to/photo.jpg")

# Remote image
# image_msg = Message.online_image("https://example.com/chart.png")

# Local audio (base64-encoded automatically)
# audio_msg = Message.local_audio("/path/to/clip.mp3")

# Remote video
# video_msg = Message.online_video("https://example.com/demo.mp4")
```

## Context — the mutable message window

`Context` holds the live list of messages for one interaction.
It does not own the LLM; it is just the conversation state.

```python {name=ch02_make_ctx}
def make_ctx(*msgs: Message) -> Context:
    return Context(list(msgs))
```

### Appending and prepending

```python {name=ch02_test_append}
def test_append_and_prepend():
    ctx = make_ctx(Message.user("hi"))
    ctx.append(Message.assistant("hello"))
    ctx.prepend(Message.system("Be kind."))

    assert ctx.messages[0].role == "system"
    assert ctx.messages[-1].role == "assistant"
    assert len(ctx.messages) == 3

def test_append_string():
    ctx = make_ctx(Message.user("hi"))
    ctx.append("Extra instruction.")
    assert ctx.messages[-1].role == "system"
```

Plain strings passed to `append` / `prepend` are wrapped in a system message.

### clone — a durable copy

`clone()` returns a new `Context` with an independent copy of the message list.
Changes to the clone do not affect the original.

```python {name=ch02_test_clone}
def test_clone_is_independent():
    original = make_ctx(Message.user("hello"))
    copy = original.clone()
    copy.append(Message.assistant("world"))

    assert len(original.messages) == 1
    assert len(copy.messages) == 2
```

### fork — a temporary scope

`fork()` is a context manager for "what-if" speculation.
All mutations inside the `with` block are discarded on exit.

```python {name=ch02_test_fork}
def test_fork_discards_changes():
    ctx = make_ctx(Message.user("start"))
    with ctx.fork():
        ctx.append(Message.assistant("ephemeral"))
        assert len(ctx.messages) == 2

    assert len(ctx.messages) == 1  # rolled back
```

### atomic — rollback on failure

`atomic()` commits mutations only if no exception is raised.
Use this when a sequence of context mutations must succeed together.

```python {name=ch02_test_atomic}
def test_atomic_rolls_back_on_error():
    ctx = make_ctx(Message.user("start"))
    try:
        with ctx.atomic():
            ctx.append(Message.assistant("partial"))
            raise ValueError("oops")
    except ValueError:
        pass
    assert len(ctx.messages) == 1  # rolled back

def test_atomic_commits_on_success():
    ctx = make_ctx(Message.user("start"))
    with ctx.atomic():
        ctx.append(Message.assistant("committed"))
    assert len(ctx.messages) == 2  # kept
```

## Test file

```python {export=tests/test_ch02.py}
from lingo import Message
from lingo.context import Context

<<ch02_make_ctx>>
<<ch02_test_append>>
<<ch02_test_clone>>
<<ch02_test_fork>>
<<ch02_test_atomic>>
```
