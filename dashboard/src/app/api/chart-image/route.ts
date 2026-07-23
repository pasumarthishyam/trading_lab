import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET(request: NextRequest) {
    const { searchParams } = new URL(request.url);
    const filePath = searchParams.get("path");

    if (!filePath) {
        return NextResponse.json({ error: "path required" }, { status: 400 });
    }

    // Resolve relative to the VCF results directory
    const resultsDir = path.join(
        process.cwd(),
        "..",
        "strategies",
        "VCF",
        "results"
    );
    const fullPath = path.join(resultsDir, filePath);

    // Security: ensure the path stays within results directory
    const resolvedPath = path.resolve(fullPath);
    const resolvedResults = path.resolve(resultsDir);
    if (!resolvedPath.startsWith(resolvedResults)) {
        return NextResponse.json({ error: "invalid path" }, { status: 403 });
    }

    if (!fs.existsSync(resolvedPath)) {
        return NextResponse.json({ error: "not found" }, { status: 404 });
    }

    const fileBuffer = fs.readFileSync(resolvedPath);

    const ext = path.extname(resolvedPath).toLowerCase();
    const contentType =
        ext === ".png"
            ? "image/png"
            : ext === ".jpg" || ext === ".jpeg"
                ? "image/jpeg"
                : ext === ".svg"
                    ? "image/svg+xml"
                    : "application/octet-stream";

    return new NextResponse(fileBuffer, {
        headers: {
            "Content-Type": contentType,
            "Cache-Control": "public, max-age=3600",
        },
    });
}
