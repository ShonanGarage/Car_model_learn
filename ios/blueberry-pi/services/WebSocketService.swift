//
//  WebSocketService.swift
//  blueberry-pi
//
//  Created by kazuki fujikawa on 2025/12/10.
//

import Foundation
import Combine

// WebSocket経由でラジコン用コマンドを送るサービス
final class WebSocketService: ObservableObject {
    @Published var isConnected = false
    @Published var lastResponse = "未接続"
    @Published var isSending = false
    
    private var task: URLSessionWebSocketTask?
    private let url = URL(string: "ws://172.20.10.3:8000/ws")!
    private let session = URLSession(configuration: .default)
    private let logPrefix = "[WebSocketService]"

    enum WsAction: String, Codable {
        case moveForward = "MOVE_FORWARD"
        case moveBackward = "MOVE_BACKWARD"
        case stop = "STOP"
        case steerLeft = "STEER_LEFT"
        case steerRight = "STEER_RIGHT"
        case resetSteer = "RESET_STEER"
        case quit = "QUIT"
    }

    private struct WsCommand: Encodable {
        let action: WsAction
        let step: Int?
    }
    
    func connectIfNeeded() {
        guard task == nil else { return }
        log("connectIfNeeded: starting connection to \(url.absoluteString)")
        let newTask = session.webSocketTask(with: url)
        task = newTask
        newTask.resume()
        isConnected = true
        log("connectIfNeeded: task resumed, isConnected=true")
        listen()
    }
    
    func disconnect() {
        log("disconnect: closing socket")
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        isConnected = false
        log("disconnect: task cleared, isConnected=false")
    }
    
    func send(action: WsAction, step: Int? = nil) {
        guard let task else {
            lastResponse = "未接続です"
            log("send: failed, no task. action=\(action.rawValue) step=\(step?.description ?? "nil")")
            return
        }
        
        isSending = true
        let command = WsCommand(action: action, step: step)
        guard let data = try? JSONEncoder().encode(command),
              let payload = String(data: data, encoding: .utf8)
        else {
            isSending = false
            lastResponse = "送信失敗: エンコードエラー"
            log("send: encode error. action=\(action.rawValue) step=\(step?.description ?? "nil")")
            return
        }
        log("send: payload=\(payload)")
        
        task.send(.string(payload)) { [weak self] error in
            DispatchQueue.main.async {
                self?.isSending = false
                if let error {
                    self?.lastResponse = "送信失敗: \(error.localizedDescription)"
                    self?.log("send: error=\(error.localizedDescription)")
                } else {
                    self?.log("send: success")
                }
            }
        }
    }
    
    private func listen() {
        log("listen: waiting for message")
        task?.receive { [weak self] result in
            DispatchQueue.main.async {
                guard let self else { return }
                switch result {
                case .failure(let error):
                    self.lastResponse = "受信エラー: \(error.localizedDescription)"
                    self.isConnected = false
                    self.task = nil
                    self.log("listen: failure=\(error.localizedDescription)")
                case .success(let message):
                    switch message {
                    case .string(let text):
                        self.lastResponse = self.humanReadableMessage(from: text)
                        self.log("listen: received string=\(text)")
                    case .data(let data):
                        let text = String(data: data, encoding: .utf8)
                        self.lastResponse = self.humanReadableMessage(from: text)
                        self.log("listen: received data=\(text ?? "<binary>")")
                    @unknown default:
                        self.lastResponse = "未知のメッセージ"
                        self.log("listen: received unknown message")
                    }
                    // 続けて待ち受け
                    self.listen()
                }
            }
        }
    }

    private func humanReadableMessage(from text: String?) -> String {
        guard let text, let data = text.data(using: .utf8) else {
            return "データ受信"
        }
        guard
            let object = try? JSONSerialization.jsonObject(with: data),
            let dict = object as? [String: Any]
        else {
            return text
        }
        if let type = dict["type"] as? String, type == "telemetry" {
            let state = dict["state"] as? String ?? "-"
            let steer = dict["steer_us"] as? Int ?? 0
            let throttle = dict["throttle_us"] as? Int ?? 0
            return "telemetry state=\(state) steer=\(steer) throttle=\(throttle)"
        }
        return text
    }

    private func log(_ message: String) {
        let ts = ISO8601DateFormatter().string(from: Date())
        print("\(logPrefix) \(ts) \(message)")
    }
}
