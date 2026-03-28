import { NextRequest, NextResponse } from "next/server";

const NOMINATIM_REVERSE = "https://nominatim.openstreetmap.org/reverse";

function pickAddressField(
  addr: Record<string, string>,
  keys: readonly string[]
): string | null {
  for (const k of keys) {
    const v = addr[k];
    if (typeof v === "string" && v.trim()) return v.trim();
  }
  return null;
}

/** Prefer a city-like name; strip trailing "County" when that’s the best we have. */
function localityFromAddress(addr: Record<string, string>): string | null {
  const primary = pickAddressField(addr, [
    "city",
    "town",
    "village",
    "municipality",
    "hamlet",
  ]);
  if (primary) return primary;

  const secondary = pickAddressField(addr, ["suburb", "neighbourhood", "quarter"]);
  if (secondary) return secondary;

  const county = pickAddressField(addr, ["county"]);
  if (county) {
    const stripped = county.replace(/\s+County$/i, "").trim();
    return stripped || county;
  }

  return pickAddressField(addr, ["state", "region"]);
}

/**
 * Reverse geocode lat/lon → display locality (e.g. "San Francisco").
 * Proxies OpenStreetMap Nominatim with a proper User-Agent (required by their usage policy).
 */
export async function GET(req: NextRequest) {
  const lat = req.nextUrl.searchParams.get("lat");
  const lon = req.nextUrl.searchParams.get("lon");
  if (lat === null || lon === null) {
    return NextResponse.json({ error: "lat and lon required" }, { status: 400 });
  }

  const latN = Number(lat);
  const lonN = Number(lon);
  if (!Number.isFinite(latN) || !Number.isFinite(lonN)) {
    return NextResponse.json({ error: "invalid coordinates" }, { status: 400 });
  }
  if (latN < -90 || latN > 90 || lonN < -180 || lonN > 180) {
    return NextResponse.json({ error: "coordinates out of range" }, { status: 400 });
  }

  const u = new URL(NOMINATIM_REVERSE);
  u.searchParams.set("lat", String(latN));
  u.searchParams.set("lon", String(lonN));
  u.searchParams.set("format", "json");
  u.searchParams.set("addressdetails", "1");

  let res: Response;
  try {
    res = await fetch(u.toString(), {
      headers: {
        Accept: "application/json",
        "Accept-Language": "en",
        // Nominatim policy: identify the application
        "User-Agent": "remop-digital-twin/1.0 (perception UI; contact: local)",
      },
      cache: "no-store",
    });
  } catch {
    return NextResponse.json({ error: "geocoder unreachable" }, { status: 502 });
  }

  if (!res.ok) {
    return NextResponse.json({ error: "geocoder failed" }, { status: 502 });
  }

  const data = (await res.json()) as { address?: Record<string, string> };
  const addr = data.address;
  if (!addr) {
    return NextResponse.json({ label: null });
  }

  const label = localityFromAddress(addr);
  return NextResponse.json({ label: label ?? null });
}
