//
//  RemoteControlViewModel.swift
//  blueberry-pi
//
//  Created by kazuki fujikawa on 2025/12/10.
//

import SwiftUI
import Combine

final class RemoteControlViewModel: ObservableObject {
    @Published var currentCommand: WebSocketService.WsAction = .stop
    @Published var pulseAnimation: Bool = false
    
    private var holdTimer: AnyCancellable?
    private let webSocketService: WebSocketService
    private var lastVector: CGSize = .zero
    private var lastStepSent: Int? = nil
    
    // スティックの設定
    let stickRadius: CGFloat = 60
    let baseRadius: CGFloat = 110
    // 方向判定の閾値（軸ベース）
    let axisThreshold: CGFloat = 28
    
    init(webSocketService: WebSocketService) {
        self.webSocketService = webSocketService
    }
    
    func impact(style: UIImpactFeedbackGenerator.FeedbackStyle = .medium) {
        let generator = UIImpactFeedbackGenerator(style: style)
        generator.impactOccurred()
    }
    
    // ドラッグ処理
    func handleDragChanged(_ translation: CGSize) {
        // 半径内にクランプ
        let vector = clamp(translation, maxRadius: baseRadius - stickRadius)
        lastVector = vector
        
        let newCommand = direction(for: vector)
        if newCommand == currentCommand {
            sendSteerUpdateIfNeeded(for: newCommand)
            return
        }
        updateCommand(newCommand)
    }
    
    func handleDragEnded() {
        updateCommand(.stop)
    }
    
    private func clamp(_ size: CGSize, maxRadius: CGFloat) -> CGSize {
        let dx = size.width
        let dy = size.height
        let distance = sqrt(dx*dx + dy*dy)
        guard distance > maxRadius else { return size }
        let scale = maxRadius / distance
        return CGSize(width: dx * scale, height: dy * scale)
    }
    
    private func direction(for vector: CGSize) -> WebSocketService.WsAction {
        let x = vector.width
        let y = vector.height
        if abs(x) < axisThreshold && abs(y) < axisThreshold {
            return .stop
        }
        // 主となる軸方向で判定（指の向きに忠実）
        if abs(x) > abs(y) {
            return x > 0 ? .steerRight : .steerLeft
        } else {
            // yは上がマイナスなので反転
            return y < 0 ? .moveForward : .moveBackward
        }
    }
    
    // コマンド更新と長押し送信
    private func updateCommand(_ newCommand: WebSocketService.WsAction) {
        guard newCommand != currentCommand else { return }
        currentCommand = newCommand
        lastStepSent = nil
        
        holdTimer?.cancel()
        
        // 接続時のみ送信
        if webSocketService.isConnected {
            webSocketService.send(action: newCommand, step: step(for: newCommand, vector: lastVector))
            impact(style: .light)
            
            guard newCommand != .stop else { return }
            holdTimer = Timer.publish(every: 0.35, on: .main, in: .common)
                .autoconnect()
                .sink { [weak self] _ in
                    guard let self = self, self.webSocketService.isConnected else { return }
                    self.webSocketService.send(action: newCommand, step: self.step(for: newCommand, vector: self.lastVector))
                }
        }
    }

    private func step(for command: WebSocketService.WsAction, vector: CGSize) -> Int? {
        switch command {
        case .steerLeft, .steerRight:
            let maxRadius = baseRadius - stickRadius
            let mag = min(abs(vector.width), maxRadius)
            let ratio = maxRadius <= 0 ? 0 : mag / maxRadius
            let minStep: CGFloat = 30
            let maxStep: CGFloat = 180
            let step = minStep + (maxStep - minStep) * ratio
            return Int(step.rounded())
        default:
            return nil
        }
    }

    private func sendSteerUpdateIfNeeded(for command: WebSocketService.WsAction) {
        guard webSocketService.isConnected else { return }
        guard command == .steerLeft || command == .steerRight else { return }
        guard let step = step(for: command, vector: lastVector) else { return }
        if let last = lastStepSent, abs(step - last) < 10 {
            return
        }
        lastStepSent = step
        webSocketService.send(action: command, step: step)
    }
    
    func onAppear() {
        webSocketService.connectIfNeeded()
        pulseAnimation = true
    }
    
    func onDisappear() {
        holdTimer?.cancel()
    }
}
