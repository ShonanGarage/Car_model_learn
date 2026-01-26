//
//  Crosshair.swift
//  blueberry-pi
//
//  Created by kazuki fujikawa on 2025/12/10.
//

import SwiftUI

// 十字ガイド用シェイプ
struct Crosshair: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        let center = CGPoint(x: rect.midX, y: rect.midY)
        path.move(to: CGPoint(x: center.x, y: rect.minY))
        path.addLine(to: CGPoint(x: center.x, y: rect.maxY))
        path.move(to: CGPoint(x: rect.minX, y: center.y))
        path.addLine(to: CGPoint(x: rect.maxX, y: center.y))
        return path
    }
}

