"use strict";

const elements = {
  appName: document.querySelector("#app-name"),
  appDescription: document.querySelector("#app-description"),
  composer: document.querySelector("#composer"),
  connectionStatus: document.querySelector("#connection-status"),
  conversation: document.querySelector("#conversation"),
  emptyState: document.querySelector("#empty-state"),
  errorBanner: document.querySelector("#error-banner"),
  errorMessage: document.querySelector("#error-message"),
  input: document.querySelector("#message-input"),
  reconnectButton: document.querySelector("#reconnect-button"),
  resetButton: document.querySelector("#reset-button"),
  sendButton: document.querySelector("#send-button"),
  sessionMode: document.querySelector("#session-mode"),
  singletonNotice: document.querySelector("#singleton-notice"),
  statusDot: document.querySelector("#status-dot"),
  themeButton: document.querySelector("#theme-button"),
};

const state = {
  socket: null,
  connected: false,
  busy: false,
  streamingMessage: null,
  turnStartedAt: null,
  reasoningTimer: null,
  toolRows: new Map(),
};

function websocketUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws`;
}

function setConnection(status, tone = "connecting") {
  elements.connectionStatus.textContent = status;
  elements.statusDot.className = `status-dot ${tone}`;
}

function updateControls() {
  const enabled = state.connected && !state.busy;
  elements.input.disabled = !enabled;
  elements.sendButton.disabled = !enabled || !elements.input.value.trim();
  elements.resetButton.disabled = !enabled;
}

function hideError() {
  elements.errorBanner.hidden = true;
  elements.reconnectButton.hidden = true;
}

function showError(message, reconnect = false) {
  elements.errorMessage.textContent = message;
  elements.errorBanner.hidden = false;
  elements.reconnectButton.hidden = !reconnect;
}

function scrollToLatest() {
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
}

function revealConversation() {
  elements.emptyState.hidden = true;
}

function stringifyContent(content) {
  return typeof content === "string" ? content : JSON.stringify(content, null, 2);
}

function appendDetail(container, title, value) {
  const block = document.createElement("div");
  block.className = "detail-block";
  const heading = document.createElement("strong");
  heading.textContent = title;
  const content = document.createElement("pre");
  content.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  block.append(heading, content);
  container.append(block);
}

function addMetadata(card, message) {
  const values = [
    ["Reasoning", message.thinking],
    ["Tool calls", message.tool_calls],
    ["Tool call ID", message.tool_call_id],
    ["Stop reason", message.stop_reason],
    ["Token usage", message.usage],
  ].filter(([, value]) => value !== undefined && value !== null);

  if (!values.length) return;

  const details = document.createElement("details");
  details.className = "message-details";
  const summary = document.createElement("summary");
  summary.textContent = "Inspect metadata";
  const body = document.createElement("div");
  body.className = "details-body";
  values.forEach(([title, value]) => appendDetail(body, title, value));
  details.append(summary, body);
  card.append(details);
}

function createMessage(role, content, streaming = false) {
  revealConversation();
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}${streaming ? " streaming" : ""}`;
  const card = document.createElement("div");
  card.className = "message-card";
  const label = document.createElement("p");
  label.className = "message-label";
  label.textContent = role;
  const text = document.createElement("p");
  text.className = "message-content";
  text.textContent = stringifyContent(content);
  card.append(label, text);
  wrapper.append(card);
  elements.conversation.append(wrapper);
  scrollToLatest();
  return { wrapper, card, text };
}

function ensureStreamingMessage() {
  if (!state.streamingMessage) {
    state.streamingMessage = createMessage("assistant", "", true);
  }
  return state.streamingMessage;
}

function reasoningElapsed() {
  const startedAt = state.turnStartedAt ?? performance.now();
  return ((performance.now() - startedAt) / 1000).toFixed(1);
}

function clearReasoningTimer() {
  if (state.reasoningTimer !== null) {
    window.clearInterval(state.reasoningTimer);
    state.reasoningTimer = null;
  }
}

function startReasoningIndicator(view) {
  if (view.reasoningStatus) return;

  const status = document.createElement("p");
  status.className = "reasoning-status";
  status.setAttribute("role", "timer");
  status.setAttribute("aria-live", "off");
  view.card.insertBefore(status, view.text);
  view.reasoningStatus = status;

  const update = () => {
    status.textContent = `Reasoning… ${reasoningElapsed()}s`;
  };
  update();
  state.reasoningTimer = window.setInterval(update, 250);
}

function finishReasoningIndicator(view) {
  if (!view?.reasoningStatus || view.reasoningComplete) return;
  clearReasoningTimer();
  view.reasoningComplete = true;
  view.reasoningStatus.classList.add("complete");
  view.reasoningStatus.textContent = `Reasoned for ${reasoningElapsed()}s`;
}

function finalizeMessage(message) {
  let view = state.streamingMessage;
  if (!view) {
    view = createMessage(message.role, message.content);
  } else {
    view.wrapper.classList.remove("streaming");
    view.wrapper.classList.remove("assistant");
    view.wrapper.classList.add(message.role);
    view.text.textContent = stringifyContent(message.content);
  }
  finishReasoningIndicator(view);
  addMetadata(view.card, message);
  state.streamingMessage = null;
  scrollToLatest();
}

function addEventRow(text, key = null) {
  revealConversation();
  const row = document.createElement("div");
  row.className = "event-row";
  row.textContent = text;
  elements.conversation.append(row);
  if (key) state.toolRows.set(key, row);
  scrollToLatest();
  return row;
}

function handleReady(event) {
  state.connected = true;
  state.busy = false;
  elements.appName.textContent = event.app.name;
  elements.appDescription.textContent = event.app.description;
  elements.sessionMode.textContent = `${event.mode} session`;
  elements.singletonNotice.hidden = event.mode !== "singleton";
  setConnection("Connected", "connected");
  hideError();
  updateControls();
  elements.input.focus();
}

function handleEvent(event) {
  switch (event.type) {
    case "ready":
      handleReady(event);
      break;
    case "turn_started":
      state.busy = true;
      state.turnStartedAt = performance.now();
      hideError();
      updateControls();
      break;
    case "token": {
      const view = ensureStreamingMessage();
      finishReasoningIndicator(view);
      view.text.textContent += event.text;
      scrollToLatest();
      break;
    }
    case "reasoning_token": {
      const view = ensureStreamingMessage();
      view.reasoning = (view.reasoning || "") + event.text;
      startReasoningIndicator(view);
      break;
    }
    case "tool_call_started":
      finishReasoningIndicator(state.streamingMessage);
      addEventRow(`Tool call · ${event.name}`, event.id);
      break;
    case "tool_call_delta": {
      const row = state.toolRows.get(event.id);
      if (row) row.textContent = `Tool arguments · ${event.arguments}`;
      break;
    }
    case "tool_call_completed": {
      const row = state.toolRows.get(event.id);
      const content = `Tool call ready · ${JSON.stringify(event.arguments)}`;
      if (row) row.textContent = content;
      else addEventRow(content);
      break;
    }
    case "message":
      if (state.streamingMessage?.reasoning && !event.message.thinking) {
        event.message.thinking = state.streamingMessage.reasoning;
      }
      finalizeMessage(event.message);
      break;
    case "structured_output":
      addEventRow(`Structured output · ${JSON.stringify(event.value)}`);
      break;
    case "turn_completed":
      state.busy = false;
      if (state.streamingMessage) finalizeMessage(event.message);
      state.turnStartedAt = null;
      updateControls();
      elements.input.focus();
      break;
    case "reset_completed":
      clearReasoningTimer();
      state.busy = false;
      state.streamingMessage = null;
      state.turnStartedAt = null;
      state.toolRows.clear();
      elements.conversation.replaceChildren(elements.emptyState);
      elements.emptyState.hidden = false;
      elements.appName.textContent = event.app.name;
      elements.appDescription.textContent = event.app.description;
      updateControls();
      elements.input.focus();
      break;
    case "error":
      clearReasoningTimer();
      state.busy = false;
      state.turnStartedAt = null;
      if (state.streamingMessage) {
        state.streamingMessage.wrapper.classList.remove("streaming");
        state.streamingMessage = null;
      }
      showError(event.message, !event.recoverable);
      updateControls();
      break;
    default:
      showError("The server sent an unsupported event.");
  }
}

function connect() {
  if (state.socket) state.socket.close();
  clearReasoningTimer();
  state.connected = false;
  state.busy = false;
  state.turnStartedAt = null;
  setConnection("Connecting");
  hideError();
  updateControls();

  const socket = new WebSocket(websocketUrl());
  state.socket = socket;

  socket.addEventListener("message", ({ data }) => {
    try {
      handleEvent(JSON.parse(data));
    } catch (error) {
      console.error("Invalid Playground event", error);
      showError("The server returned an invalid event.");
    }
  });

  socket.addEventListener("close", () => {
    if (state.socket !== socket) return;
    clearReasoningTimer();
    state.connected = false;
    state.busy = false;
    state.turnStartedAt = null;
    setConnection("Disconnected", "disconnected");
    showError("Connection to the local Playground was closed.", true);
    updateControls();
  });

  socket.addEventListener("error", () => {
    showError("Could not connect to the local Playground.", true);
  });
}

function resizeInput() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 180)}px`;
}

function sendMessage() {
  const content = elements.input.value.trim();
  if (!content || !state.connected || state.busy) return;
  createMessage("user", content);
  state.socket.send(JSON.stringify({ type: "user_message", content }));
  elements.input.value = "";
  resizeInput();
  updateControls();
}

elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage();
});

elements.input.addEventListener("input", () => {
  resizeInput();
  updateControls();
});

elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

elements.resetButton.addEventListener("click", () => {
  if (!state.connected || state.busy) return;
  state.busy = true;
  updateControls();
  state.socket.send(JSON.stringify({ type: "reset" }));
});

elements.reconnectButton.addEventListener("click", connect);

elements.themeButton.addEventListener("click", () => {
  const current = document.documentElement.dataset.theme;
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("lingo-playground-theme", next);
});

const storedTheme = localStorage.getItem("lingo-playground-theme");
if (storedTheme === "dark" || storedTheme === "light") {
  document.documentElement.dataset.theme = storedTheme;
} else if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
  document.documentElement.dataset.theme = "dark";
}

connect();
