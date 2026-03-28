import CameraOverlay from "./components/CameraOverlay";

export default function Home() {
  return (
    <div className="flex min-h-full flex-col items-center bg-zinc-50 px-4 py-8 dark:bg-black">
      <main className="flex w-full max-w-4xl flex-col gap-6">
        <div>
          <h1 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-50">
            remop — live detection and depth
          </h1>
          <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
            Start the inference server:{" "}
            <code className="rounded bg-zinc-200 px-1 font-mono text-xs dark:bg-zinc-800">
              cd backend && uvicorn inference_server:app --host 0.0.0.0 --port 8000
            </code>
            . Then start the camera below (requires user gesture in Safari).
          </p>
        </div>
        <CameraOverlay />
      </main>
    </div>
  );
}
