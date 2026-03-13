from lingo.llm import Message, ImageContent, AudioContent


def test_online_image_message():
    """Tests creating a message with an online image URL."""
    url = "https://example.com/image.png"
    msg = Message.online_image(url, detail="high")

    assert msg.role == "user"
    assert isinstance(msg.content, ImageContent)
    assert msg.content.image_url["url"] == url
    assert msg.content.image_url["detail"] == "high"


def test_online_video_message():
    """Tests creating a message with an online video URL."""
    url = "https://example.com/video.mp4"
    msg = Message.online_video(url)

    assert msg.role == "user"
    assert msg.content.video_url["url"] == url


def test_local_image_message(tmp_path):
    """Tests creating a message with a local image file."""
    path = tmp_path / "dummy.png"
    path.write_text("fake image data")

    msg = Message.local_image(str(path))

    assert msg.role == "user"
    assert isinstance(msg.content, ImageContent)
    assert "data:image/png;base64," in msg.content.image_url["url"]


def test_local_audio_message(tmp_path):
    """Tests creating a message with a local audio file."""
    path = tmp_path / "dummy.mp3"
    path.write_text("fake audio data")

    msg = Message.local_audio(str(path))

    assert msg.role == "user"
    assert isinstance(msg.content, AudioContent)
    assert msg.content.input_audio["format"] == "mp3"


def test_message_model_dump_text():
    """Tests model_dump for a standard text message."""
    msg = Message.user("Hello world")
    dump = msg.model_dump()

    assert dump["role"] == "user"
    assert dump["content"] == "Hello world"


def test_message_model_dump_multimodal():
    """Tests model_dump for a multimodal message."""
    url = "https://example.com/image.png"
    msg = Message.online_image(url)
    dump = msg.model_dump()

    assert dump["role"] == "user"
    assert isinstance(dump["content"], list)
    assert dump["content"][0]["type"] == "image_url"
    assert dump["content"][0]["image_url"]["url"] == url
