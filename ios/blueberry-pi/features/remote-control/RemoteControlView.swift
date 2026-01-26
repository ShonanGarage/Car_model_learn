//
//  RemoteControlView.swift
//  blueberry-pi
//
//  Created by kazuki fujikawa on 2025/12/10.
//

import SwiftUI

// ラジコン用UI
struct RemoteControlView: View {
    @StateObject private var webSocketService = WebSocketService()
    @StateObject private var viewModel: RemoteControlViewModel
    @State private var stickOffset: CGSize = .zero
    
    init() {
        let service = WebSocketService()
        _webSocketService = StateObject(wrappedValue: service)
        _viewModel = StateObject(wrappedValue: RemoteControlViewModel(webSocketService: service))
    }
    
    var body: some View {
        ZStack {
            // 背景グラデーション
            LinearGradient(
                gradient: Gradient(colors: [
                    Color(red: 0.05, green: 0.05, blue: 0.12),
                    Color(red: 0.02, green: 0.02, blue: 0.08),
                    Color.black
                ]),
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
            
            // アニメーション背景エフェクト
            if webSocketService.isConnected {
                Circle()
                    .fill(
                        RadialGradient(
                            gradient: Gradient(colors: [
                                Color.cyan.opacity(0.15),
                                Color.blue.opacity(0.05),
                                Color.clear
                            ]),
                            center: .center,
                            startRadius: 50,
                            endRadius: 300
                        )
                    )
                    .frame(width: 600, height: 600)
                    .offset(y: -200)
                    .blur(radius: 60)
                    .opacity(viewModel.pulseAnimation ? 0.6 : 0.3)
                    .animation(
                        Animation.easeInOut(duration: 2.0)
                            .repeatForever(autoreverses: true),
                        value: viewModel.pulseAnimation
                    )
            }
            
            VStack(spacing: 0) {
                // 接続状態インジケーター
                HStack(spacing: 12) {
                    Circle()
                        .fill(webSocketService.isConnected ? Color.cyan : Color.red.opacity(0.6))
                        .frame(width: 12, height: 12)
                        .shadow(
                            color: webSocketService.isConnected ? Color.cyan.opacity(0.8) : Color.red.opacity(0.6),
                            radius: webSocketService.isConnected ? 8 : 4
                        )
                        .scaleEffect(viewModel.pulseAnimation ? 1.2 : 1.0)
                        .animation(
                            Animation.easeInOut(duration: 1.5)
                                .repeatForever(autoreverses: true),
                            value: viewModel.pulseAnimation
                        )
                    
                    Text(webSocketService.isConnected ? "ONLINE" : "OFFLINE")
                        .font(.system(size: 13, weight: .bold, design: .rounded))
                        .foregroundColor(webSocketService.isConnected ? Color.cyan : Color.red.opacity(0.8))
                        .tracking(2)
                }
                .padding(.top, 60)
                .padding(.bottom, 40)
                
                // スティック UI
                VStack(spacing: 12) {
                    ZStack {
                        // 固定要素（同じレイヤーに順番に配置）
                        // 外側のグローリング
                        if webSocketService.isConnected && viewModel.currentCommand != .stop {
                            Circle()
                                .stroke(
                                    LinearGradient(
                                        gradient: Gradient(colors: [
                                            Color.cyan.opacity(0.6),
                                            Color.blue.opacity(0.3),
                                            Color.clear
                                        ]),
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    ),
                                    lineWidth: 3
                                )
                                .frame(width: viewModel.baseRadius * 2 + 20, height: viewModel.baseRadius * 2 + 20)
                                .blur(radius: 8)
                                .opacity(viewModel.pulseAnimation ? 0.8 : 0.4)
                                .animation(
                                    Animation.easeInOut(duration: 1.0)
                                        .repeatForever(autoreverses: true),
                                    value: viewModel.pulseAnimation
                                )
                                .allowsHitTesting(false)
                        }
                        
                        // ベースサークル
                        Circle()
                            .fill(
                                LinearGradient(
                                    gradient: Gradient(colors: [
                                        Color(white: 0.18),
                                        Color(white: 0.10),
                                        Color(white: 0.08)
                                    ]),
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .frame(width: viewModel.baseRadius * 2, height: viewModel.baseRadius * 2)
                            .shadow(color: .black.opacity(0.6), radius: 20, x: 0, y: 10)
                            .overlay(
                                Circle()
                                    .stroke(
                                        LinearGradient(
                                            gradient: Gradient(colors: [
                                                Color.white.opacity(0.2),
                                                Color.white.opacity(0.05)
                                            ]),
                                            startPoint: .topLeading,
                                            endPoint: .bottomTrailing
                                        ),
                                        lineWidth: 1.5
                                    )
                            )
                            .allowsHitTesting(false)
                        
                        // ガイドライン
                        Crosshair()
                            .stroke(
                                LinearGradient(
                                    gradient: Gradient(colors: [
                                        Color.cyan.opacity(0.3),
                                        Color.blue.opacity(0.2),
                                        Color.cyan.opacity(0.3)
                                    ]),
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                lineWidth: 1.5
                            )
                            .frame(width: viewModel.baseRadius * 1.6, height: viewModel.baseRadius * 1.6)
                            .blur(radius: 0.5)
                            .allowsHitTesting(false)
                        
                        // スティック（動く要素のみ）
                        ZStack {
                            // グロー効果
                            if webSocketService.isConnected && viewModel.currentCommand != .stop {
                                Circle()
                                    .fill(
                                        RadialGradient(
                                            gradient: Gradient(colors: [
                                                Color.cyan.opacity(0.4),
                                                Color.clear
                                            ]),
                                            center: .center,
                                            startRadius: 0,
                                            endRadius: viewModel.stickRadius
                                        )
                                    )
                                    .frame(width: viewModel.stickRadius * 2.5, height: viewModel.stickRadius * 2.5)
                                    .blur(radius: 15)
                            }
                            
                            Circle()
                                .fill(
                                    LinearGradient(
                                        gradient: Gradient(colors: [
                                            Color.white.opacity(0.95),
                                            Color(white: 0.75),
                                            Color(white: 0.65)
                                        ]),
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                )
                                .frame(width: viewModel.stickRadius * 2, height: viewModel.stickRadius * 2)
                                .shadow(color: .black.opacity(0.5), radius: 12, x: 0, y: 6)
                                .overlay(
                                    Circle()
                                        .stroke(
                                            LinearGradient(
                                                gradient: Gradient(colors: [
                                                    Color.white.opacity(0.8),
                                                    Color.white.opacity(0.3)
                                                ]),
                                                startPoint: .topLeading,
                                                endPoint: .bottomTrailing
                                            ),
                                            lineWidth: 2
                                        )
                                )
                                .overlay(
                                    Circle()
                                        .fill(
                                            RadialGradient(
                                                gradient: Gradient(colors: [
                                                    Color.white.opacity(0.6),
                                                    Color.clear
                                                ]),
                                                center: UnitPoint(x: 0.3, y: 0.3),
                                                startRadius: 0,
                                                endRadius: viewModel.stickRadius * 0.8
                                            )
                                        )
                                )
                        }
                        .offset(stickOffset)
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { value in
                                    let maxRadius = viewModel.baseRadius - viewModel.stickRadius
                                    let dx = value.translation.width
                                    let dy = value.translation.height
                                    let distance = sqrt(dx*dx + dy*dy)
                                    
                                    if distance > maxRadius {
                                        let scale = maxRadius / distance
                                        stickOffset = CGSize(width: dx * scale, height: dy * scale)
                                    } else {
                                        stickOffset = value.translation
                                    }
                                    
                                    viewModel.handleDragChanged(stickOffset)
                                }
                                .onEnded { _ in
                                    withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                                        stickOffset = .zero
                                    }
                                    viewModel.handleDragEnded()
                                }
                        )
                    }
                    
                    // コマンド表示
                    if viewModel.currentCommand != .stop {
                        Text(commandName(viewModel.currentCommand))
                            .font(.system(size: 14, weight: .semibold, design: .rounded))
                            .foregroundColor(.cyan)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(
                                Capsule()
                                    .fill(Color.black.opacity(0.6))
                                    .overlay(
                                        Capsule()
                                            .stroke(
                                                LinearGradient(
                                                    gradient: Gradient(colors: [
                                                        Color.cyan.opacity(0.6),
                                                        Color.blue.opacity(0.4)
                                                    ]),
                                                    startPoint: .leading,
                                                    endPoint: .trailing
                                                ),
                                                lineWidth: 1.5
                                            )
                                    )
                            )
                            .shadow(color: Color.cyan.opacity(0.4), radius: 8, x: 0, y: 4)
                            .transition(.opacity.combined(with: .scale))
                    }
                }
                .padding(.vertical, 40)
                
                Spacer()
                
                // コントロールボタン
                HStack(spacing: 16) {
                    // 接続ボタン
                    Button {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.6)) {
                            webSocketService.connectIfNeeded()
                        }
                        viewModel.impact(style: .light)
                    } label: {
                        HStack(spacing: 8) {
                            Image(systemName: "link")
                                .font(.system(size: 16, weight: .semibold))
                            Text("接続")
                                .font(.system(size: 15, weight: .semibold, design: .rounded))
                        }
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(
                            LinearGradient(
                                gradient: Gradient(colors: [
                                    Color.cyan.opacity(0.8),
                                    Color.blue.opacity(0.7)
                                ]),
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .cornerRadius(16)
                        .shadow(color: Color.cyan.opacity(0.4), radius: 12, x: 0, y: 6)
                    }
                    .disabled(webSocketService.isConnected)
                    .opacity(webSocketService.isConnected ? 0.5 : 1.0)
                    
                    // 切断ボタン
                    Button {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.6)) {
                            webSocketService.disconnect()
                        }
                        viewModel.impact(style: .light)
                    } label: {
                        HStack(spacing: 8) {
                            Image(systemName: "xmark")
                                .font(.system(size: 16, weight: .semibold))
                            Text("切断")
                                .font(.system(size: 15, weight: .semibold, design: .rounded))
                        }
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(
                            LinearGradient(
                                gradient: Gradient(colors: [
                                    Color.red.opacity(0.7),
                                    Color.red.opacity(0.5)
                                ]),
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .cornerRadius(16)
                        .shadow(color: Color.red.opacity(0.3), radius: 12, x: 0, y: 6)
                    }
                    .disabled(!webSocketService.isConnected)
                    .opacity(webSocketService.isConnected ? 1.0 : 0.5)
                }
                .padding(.horizontal, 32)
                .padding(.bottom, 50)
            }
        }
        .onAppear {
            viewModel.onAppear()
        }
        .onDisappear {
            viewModel.onDisappear()
        }
    }
    
    // コマンド名を日本語に変換
    private func commandName(_ command: WebSocketService.WsAction) -> String {
        switch command {
        case .moveForward:
            return "前進"
        case .moveBackward:
            return "後退"
        case .steerLeft:
            return "左回転"
        case .steerRight:
            return "右回転"
        case .stop:
            return "停止"
        case .resetSteer:
            return "操舵リセット"
        case .quit:
            return "終了"
        }
    }
}
