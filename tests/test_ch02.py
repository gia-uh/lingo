from lingo import Message
from lingo.context import Context

def make_ctx(*msgs: Message) -> Context:
    return Context(list(msgs))

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

def test_clone_is_independent():
    original = make_ctx(Message.user("hello"))
    copy = original.clone()
    copy.append(Message.assistant("world"))

    assert len(original.messages) == 1
    assert len(copy.messages) == 2

def test_fork_discards_changes():
    ctx = make_ctx(Message.user("start"))
    with ctx.fork():
        ctx.append(Message.assistant("ephemeral"))
        assert len(ctx.messages) == 2

    assert len(ctx.messages) == 1  # rolled back

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

