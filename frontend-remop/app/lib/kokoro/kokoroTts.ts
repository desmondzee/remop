/**
 * Kokoro TTS for the browser — aligned with hexgrad/kokoro kokoro.js (Transformers.js + ONNX).
 * @see https://github.com/hexgrad/kokoro/tree/main/kokoro.js
 */

import {
  StyleTextToSpeech2Model,
  AutoTokenizer,
  Tensor,
  RawAudio,
  type PreTrainedTokenizer,
  type ProgressCallback,
} from "@huggingface/transformers";
import { phonemize } from "./phonemize";
import { TextSplitterStream } from "./splitter";
import { getVoiceData, VOICES } from "./voices";

const STYLE_DIM = 256;
const SAMPLE_RATE = 24000;

export type KokoroDtype = "fp32" | "fp16" | "q8" | "q4" | "q4f16";
export type KokoroDevice = "wasm" | "webgpu" | "cpu" | null;

export type GenerateOptions = {
  /** Must be a key of `VOICES` (e.g. from env); validated at runtime. */
  voice?: string;
  speed?: number;
};

export type StreamGenerateOptions = GenerateOptions & {
  split_pattern?: RegExp | null;
};

export class KokoroTTS {
  model: StyleTextToSpeech2Model;
  tokenizer: PreTrainedTokenizer;

  constructor(model: StyleTextToSpeech2Model, tokenizer: PreTrainedTokenizer) {
    this.model = model;
    this.tokenizer = tokenizer;
  }

  static async from_pretrained(
    modelId: string,
    {
      dtype = "fp32",
      device = null,
      progress_callback = null,
    }: {
      dtype?: KokoroDtype;
      device?: KokoroDevice;
      progress_callback?: ProgressCallback | null;
    } = {}
  ): Promise<KokoroTTS> {
    const model = StyleTextToSpeech2Model.from_pretrained(modelId, {
      progress_callback: progress_callback ?? undefined,
      dtype,
      device: device ?? undefined,
    });
    const tokenizer = AutoTokenizer.from_pretrained(modelId, {
      progress_callback: progress_callback ?? undefined,
    });
    const info = await Promise.all([model, tokenizer]);
    return new KokoroTTS(info[0], info[1]);
  }

  get voices(): typeof VOICES {
    return VOICES;
  }

  list_voices(): void {
    console.table(VOICES);
  }

  _validate_voice(voice: string): "a" | "b" {
    if (!Object.hasOwn(VOICES, voice)) {
      console.error(`Voice "${voice}" not found. Available voices:`);
      console.table(VOICES);
      throw new Error(
        `Voice "${voice}" not found. Should be one of: ${Object.keys(VOICES).join(", ")}.`
      );
    }
    const language = voice.at(0) as "a" | "b";
    return language;
  }

  async generate(
    text: string,
    { voice = "af_heart", speed = 1 }: GenerateOptions = {}
  ): Promise<RawAudio> {
    const language = this._validate_voice(voice);
    const phonemes = await phonemize(text, language);
    const { input_ids } = this.tokenizer(phonemes, { truncation: true });
    return this.generate_from_ids(input_ids, { voice, speed });
  }

  async generate_from_ids(
    input_ids: Tensor,
    { voice = "af_heart", speed = 1 }: GenerateOptions = {}
  ): Promise<RawAudio> {
    const numTokens = Math.min(Math.max(input_ids.dims.at(-1)! - 2, 0), 509);
    const data = await getVoiceData(voice);
    const offset = numTokens * STYLE_DIM;
    const voiceData = data.slice(offset, offset + STYLE_DIM);
    const inputs = {
      input_ids,
      style: new Tensor("float32", voiceData, [1, STYLE_DIM]),
      speed: new Tensor("float32", [speed], [1]),
    };
    const { waveform } = await this.model(inputs);
    return new RawAudio(waveform.data as Float32Array, SAMPLE_RATE);
  }

  async *stream(
    text: string | TextSplitterStream,
    { voice = "af_heart", speed = 1, split_pattern = null }: StreamGenerateOptions = {}
  ): AsyncGenerator<{
    text: string;
    phonemes: string;
    audio: RawAudio;
  }> {
    const language = this._validate_voice(voice);
    let splitter: TextSplitterStream;
    if (text instanceof TextSplitterStream) {
      splitter = text;
    } else if (typeof text === "string") {
      splitter = new TextSplitterStream();
      const chunks = split_pattern
        ? text
            .split(split_pattern)
            .map((chunk) => chunk.trim())
            .filter((chunk) => chunk.length > 0)
        : [text];
      splitter.push(...chunks);
    } else {
      throw new Error("Invalid input type. Expected string or TextSplitterStream.");
    }
    for await (const sentence of splitter) {
      const phonemes = await phonemize(sentence, language);
      const { input_ids } = this.tokenizer(phonemes, { truncation: true });
      const audio = await this.generate_from_ids(input_ids, { voice, speed });
      yield { text: sentence, phonemes, audio };
    }
  }
}

export { TextSplitterStream };
