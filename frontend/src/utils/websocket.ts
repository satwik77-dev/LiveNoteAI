import type { BackendMessage, FrontendMessage } from "../types/meeting";

export function createMeetingSocket(url: string): WebSocket {
  return new WebSocket(url);
}

export function sendSocketMessage(socket: WebSocket, message: FrontendMessage) {
  if (socket.readyState !== WebSocket.OPEN) {
    throw new Error("WebSocket is not open.");
  }

  socket.send(JSON.stringify(message));
}

export function parseSocketMessage(raw: string): BackendMessage {
  return JSON.parse(raw) as BackendMessage;
}
