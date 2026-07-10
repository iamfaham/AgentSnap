from agentsnap.adapters.anthropic import AnthropicAdapter, AnthropicRecordingStream
from agentsnap.adapters.openai import OpenAIAdapter, OpenAIRecordingStream
from agentsnap.core.recorder import AgentRecorder, TraceAccumulator, _accumulator_var


class FakeDelta:
    def __init__(self, content=None):
        self.content = content


class FakeChoice:
    def __init__(self, content=None):
        self.delta = FakeDelta(content)


class FakeUsage:
    def __init__(self, total_tokens):
        self.total_tokens = total_tokens


class FakeChunk:
    def __init__(self, content=None, usage=None):
        self.choices = [FakeChoice(content)] if content is not None else [FakeChoice(None)]
        self.usage = FakeUsage(usage) if usage is not None else None

    def model_dump(self, mode="json"):
        return {
            "content": self.choices[0].delta.content,
            "usage": {"total_tokens": self.usage.total_tokens} if self.usage else None,
        }


class EmptyChoicesChunk:
    """Chunk with an empty choices list (guard against IndexError)."""
    choices = []
    usage = None

    def model_dump(self, mode="json"):
        return {"choices": []}


class FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        self.closed = True


class FakeCompletions:
    def __init__(self, stream):
        self._stream = stream
        self.received_kwargs = None

    def create(self, **kwargs):
        self.received_kwargs = kwargs
        return self._stream


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, completions):
        self.chat = FakeChat(completions)


def _chunks():
    return [
        FakeChunk("Hello, "),
        FakeChunk("world!"),
        FakeChunk(None, usage=42),
    ]


# ── OpenAIRecordingStream direct tests ─────────────────────────────────────────

def test_chunks_reach_consumer_in_order_unmodified():
    chunks = _chunks()
    acc = TraceAccumulator()
    stream = OpenAIRecordingStream(FakeStream(chunks), [{"role": "user", "content": "hi"}], acc)
    consumed = list(stream)
    assert consumed == chunks


def test_single_llm_call_event_pushed_on_exhaustion():
    chunks = _chunks()
    acc = TraceAccumulator()
    stream = OpenAIRecordingStream(FakeStream(chunks), [{"role": "user", "content": "hi"}], acc)
    list(stream)
    assert len(acc.trace) == 1
    event = acc.trace[0]
    assert event["type"] == "llm_call"
    assert event["response"] == "Hello, world!"
    assert event["tokens"] == 42
    assert event["raw_response"]["__stream__"] is True
    assert len(event["raw_response"]["chunks"]) == 3


def test_early_close_after_partial_consumption_records_partial_text_once():
    chunks = _chunks()
    fake_stream = FakeStream(chunks)
    acc = TraceAccumulator()
    stream = OpenAIRecordingStream(fake_stream, [{"role": "user", "content": "hi"}], acc)

    it = iter(stream)
    first = next(it)
    assert first is chunks[0]

    stream.close()
    assert fake_stream.closed is True
    assert len(acc.trace) == 1
    assert acc.trace[0]["response"] == "Hello, "

    # closing again, or exhausting the (now-closed) iterator, must not double-push
    stream.close()
    assert len(acc.trace) == 1


def test_guards_empty_choices_chunk():
    acc = TraceAccumulator()
    stream = OpenAIRecordingStream(FakeStream([EmptyChoicesChunk()]), [], acc)
    list(stream)
    assert acc.trace[0]["response"] == ""
    assert acc.trace[0]["tokens"] == 0


# ── Adapter integration tests ──────────────────────────────────────────────────

def test_adapter_streams_true_forwards_stream_and_tees(tmp_path):
    completions = FakeCompletions(FakeStream(_chunks()))
    client = OpenAIAdapter(FakeClient(completions))

    with AgentRecorder("stream_rec", snapshot_dir=str(tmp_path)) as rec:
        result = client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        assert isinstance(result, OpenAIRecordingStream)
        collected = list(result)
        rec.output = "".join(
            (c.choices[0].delta.content or "") for c in collected
        )

    assert completions.received_kwargs["stream"] is True
    assert len(rec.accumulator.trace) == 1
    event = rec.accumulator.trace[0]
    assert event["response"] == "Hello, world!"
    assert event["raw_response"]["__stream__"] is True


def test_adapter_stream_false_path_unchanged_forces_stream_false(tmp_path):
    class NonStreamResponse:
        def __init__(self):
            self.choices = [type("C", (), {"message": type("M", (), {"content": "hi"})()})()]
            self.usage = type("U", (), {"total_tokens": 5})()

        def model_dump(self, mode="json"):
            return {"content": "hi"}

    completions = FakeCompletions(NonStreamResponse())
    client = OpenAIAdapter(FakeClient(completions))

    with AgentRecorder("nonstream_rec", snapshot_dir=str(tmp_path)) as rec:
        result = client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )
        rec.output = "done"

    assert completions.received_kwargs["stream"] is False
    assert not isinstance(result, OpenAIRecordingStream)
    assert rec.accumulator.trace[0]["response"] == "hi"


def test_adapter_passthrough_without_accumulator_returns_raw_stream():
    raw_stream = FakeStream(_chunks())
    completions = FakeCompletions(raw_stream)
    client = OpenAIAdapter(FakeClient(completions))

    result = client.chat.completions.create(
        model="m", messages=[], stream=True
    )
    assert result is raw_stream
    assert not isinstance(result, OpenAIRecordingStream)


# ── Anthropic fakes ─────────────────────────────────────────────────────────────

class FakeAnthUsageStart:
    def __init__(self, input_tokens):
        self.input_tokens = input_tokens


class FakeAnthMessage:
    def __init__(self, usage):
        self.usage = usage


class FakeAnthDelta:
    def __init__(self, text=None):
        self.text = text


class FakeAnthUsageDelta:
    def __init__(self, output_tokens):
        self.output_tokens = output_tokens


class FakeAnthEvent:
    def __init__(self, type, message=None, delta=None, usage=None):
        self.type = type
        self.message = message
        self.delta = delta
        self.usage = usage

    def model_dump(self, mode="json"):
        return {"type": self.type}


def _anth_events():
    return [
        FakeAnthEvent("message_start", message=FakeAnthMessage(FakeAnthUsageStart(11))),
        FakeAnthEvent("content_block_delta", delta=FakeAnthDelta("Hello, ")),
        FakeAnthEvent("content_block_delta", delta=FakeAnthDelta("world!")),
        FakeAnthEvent("message_delta", usage=FakeAnthUsageDelta(9)),
        FakeAnthEvent("message_stop"),
    ]


class FakeAnthStream:
    def __init__(self, events):
        self._events = events
        self.closed = False

    def __iter__(self):
        return iter(self._events)

    def close(self):
        self.closed = True


class FakeMessagesCreate:
    def __init__(self, stream):
        self._stream = stream
        self.received_kwargs = None

    def create(self, **kwargs):
        self.received_kwargs = kwargs
        return self._stream


class FakeAnthClient:
    def __init__(self, messages):
        self.messages = messages


# ── AnthropicRecordingStream direct tests ──────────────────────────────────────

def test_anthropic_chunks_reach_consumer_in_order_unmodified():
    events = _anth_events()
    acc = TraceAccumulator()
    stream = AnthropicRecordingStream(
        FakeAnthStream(events), [{"role": "user", "content": "hi"}], acc
    )
    consumed = list(stream)
    assert consumed == events


def test_anthropic_single_llm_call_event_pushed_on_exhaustion():
    events = _anth_events()
    acc = TraceAccumulator()
    stream = AnthropicRecordingStream(
        FakeAnthStream(events), [{"role": "user", "content": "hi"}], acc
    )
    list(stream)
    assert len(acc.trace) == 1
    event = acc.trace[0]
    assert event["type"] == "llm_call"
    assert event["response"] == "Hello, world!"
    assert event["tokens"] == 20  # 11 input + 9 output
    assert event["raw_response"]["__stream__"] is True
    assert len(event["raw_response"]["chunks"]) == 5


def test_anthropic_early_close_after_partial_consumption_records_partial_text_once():
    events = _anth_events()
    fake_stream = FakeAnthStream(events)
    acc = TraceAccumulator()
    stream = AnthropicRecordingStream(fake_stream, [{"role": "user", "content": "hi"}], acc)

    it = iter(stream)
    first = next(it)
    assert first is events[0]

    stream.close()
    assert fake_stream.closed is True
    assert len(acc.trace) == 1
    assert acc.trace[0]["response"] == ""
    assert acc.trace[0]["tokens"] == 11

    # closing again must not double-push
    stream.close()
    assert len(acc.trace) == 1


# ── Anthropic adapter integration tests ────────────────────────────────────────

def test_anthropic_adapter_streams_true_forwards_stream_and_tees(tmp_path):
    messages_proxy = FakeMessagesCreate(FakeAnthStream(_anth_events()))
    client = AnthropicAdapter(FakeAnthClient(messages_proxy))

    with AgentRecorder("anth_stream_rec", snapshot_dir=str(tmp_path)) as rec:
        result = client.messages.create(
            model="m", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        assert isinstance(result, AnthropicRecordingStream)
        list(result)
        rec.output = "done"

    assert messages_proxy.received_kwargs["stream"] is True
    assert len(rec.accumulator.trace) == 1
    event = rec.accumulator.trace[0]
    assert event["response"] == "Hello, world!"
    assert event["tokens"] == 20
    assert event["raw_response"]["__stream__"] is True


def test_anthropic_adapter_passthrough_without_accumulator_returns_raw_stream():
    raw_stream = FakeAnthStream(_anth_events())
    messages_proxy = FakeMessagesCreate(raw_stream)
    client = AnthropicAdapter(FakeAnthClient(messages_proxy))

    result = client.messages.create(model="m", messages=[], stream=True)
    assert result is raw_stream
    assert not isinstance(result, AnthropicRecordingStream)
