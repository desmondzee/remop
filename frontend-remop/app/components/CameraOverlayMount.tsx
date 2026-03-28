"use client";

import dynamic from "next/dynamic";

const CameraOverlay = dynamic(() => import("./CameraOverlay"), {
  ssr: false,
  loading: () => (
    <p className="text-sm text-zinc-500 dark:text-zinc-400">Loading controls…</p>
  ),
});

export default function CameraOverlayMount() {
  return <CameraOverlay />;
}
