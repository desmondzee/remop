import CameraOverlayMount from "./components/CameraOverlayMount";

export default function Home() {
  return (
    <div className="digital-twin-root flex h-full min-h-0 min-w-0 flex-1 flex-col">
      <main className="flex h-full min-h-0 min-w-0 w-full flex-1 flex-col">
        <CameraOverlayMount />
      </main>
    </div>
  );
}
