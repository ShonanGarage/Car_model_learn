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
        VStack(spacing: 12) {
            Image(systemName: "gamecontroller")
                .font(.system(size: 44))
            Text("Remote Control UI is temporarily disabled.")
                .font(.headline)
            Text(webSocketService.isConnected ? "Connected" : "Disconnected")
                .foregroundStyle(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black.ignoresSafeArea())
        .onAppear {
            viewModel.onAppear()
        }
        .onDisappear {
            viewModel.onDisappear()
        }
    }
}
