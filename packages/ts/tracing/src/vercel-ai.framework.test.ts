/**
 * Drive vercelAI() through real AI SDK generateText.
 * No network: MockLanguageModelV3 + mocked ingest.
 */
import { generateText, stepCountIs, streamText, tool } from "ai";
import { convertArrayToReadableStream, MockLanguageModelV3 } from "ai/test";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { ingestFetchMock, jsonBody, LEMMA_PROJECT_ID } from "../test-helpers";
import { vercelAI } from "./vercel-ai";

function mockModel(text: string) {
  return new MockLanguageModelV3({
    doGenerate: async () => ({
      finishReason: "stop",
      usage: { inputTokens: 8, outputTokens: 4, totalTokens: 12 },
      content: [{ type: "text", text }],
      warnings: [],
    }),
  });
}

describe("vercelAI through real AI SDK", () => {
  it("generateText sends one owned trace", async () => {
    const fetchMock = ingestFetchMock();
    const lemmaTelemetry = vercelAI({
      apiKey: "key",
      projectId: LEMMA_PROJECT_ID,
      fetch: fetchMock as typeof fetch,
      metadata: { threadId: "thread-1", userId: "user-1" },
    });

    const result = await generateText({
      model: mockModel("hello from vercel ai"),
      prompt: "hi",
      telemetry: {
        isEnabled: true,
        functionId: "support-agent",
        integrations: [lemmaTelemetry],
      },
    });
    await lemmaTelemetry.flush();

    expect(result.text).toBe("hello from vercel ai");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const trace = jsonBody(fetchMock.mock.calls[0]).trace;
    expect(trace).toMatchObject({
      name: "support-agent",
      thread_id: "thread-1",
      user_id: "user-1",
    });
    expect(
      trace.spans.some((span: { type: string }) => span.type === "generation"),
    ).toBe(true);
  });
});

describe("vercelAI through real AI SDK streamText failures", () => {
  const usage = {
    inputTokens: { total: 11, noCache: 11, cacheRead: 0, cacheWrite: 0 },
    outputTokens: { total: 7, text: 7, reasoning: 0 },
  };
  const finish = (unified: "stop" | "tool-calls") =>
    ({
      type: "finish",
      finishReason: { unified, raw: unified },
      usage,
    }) as const;
  const start = { type: "stream-start", warnings: [] } as const;
  const failure = new Error("429 rate limit exceeded");

  async function runStream(
    doStream: MockLanguageModelV3["doStream"],
    abortSignal?: AbortSignal,
  ) {
    const fetchMock = ingestFetchMock();
    const lemmaTelemetry = vercelAI({
      apiKey: "key",
      projectId: LEMMA_PROJECT_ID,
      fetch: fetchMock as typeof fetch,
    });
    const result = streamText({
      model: new MockLanguageModelV3({ doStream }),
      prompt: "hi",
      maxRetries: 0,
      abortSignal,
      onError: () => {},
      stopWhen: stepCountIs(4),
      tools: {
        lookup: tool({
          inputSchema: z.object({ q: z.string() }),
          execute: async () => "ok",
        }),
      },
      telemetry: {
        functionId: "support-agent",
        integrations: [lemmaTelemetry],
      },
    });
    for await (const _part of result.fullStream) {
      // drain
    }
    await lemmaTelemetry.flush();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    return jsonBody(fetchMock.mock.calls[0]).trace;
  }

  const generations = (trace: { spans: { type: string }[] }) =>
    trace.spans.filter((span) => span.type === "generation");

  it("records one generation per model call", async () => {
    const trace = await runStream(async () => ({
      stream: convertArrayToReadableStream([
        start,
        { type: "text-start", id: "1" },
        { type: "text-delta", id: "1", delta: "done" },
        { type: "text-end", id: "1" },
        finish("stop"),
      ]),
    }));
    expect(trace.error).toBeNull();
    expect(generations(trace)).toHaveLength(1);
  });

  it("sends a failed trace when the provider call throws", async () => {
    const trace = await runStream(async () => {
      throw failure;
    });
    expect(trace.error).toContain("429 rate limit exceeded");
    expect(generations(trace)[0]).toMatchObject({
      error: expect.stringContaining("429 rate limit exceeded"),
    });
  });

  it("fails the trace on a mid-stream provider error", async () => {
    const trace = await runStream(async () => ({
      stream: convertArrayToReadableStream([
        start,
        { type: "text-start", id: "1" },
        { type: "text-delta", id: "1", delta: "partial" },
        { type: "error", error: failure },
      ]),
    }));
    expect(trace.error).toContain("finishReason: error");
    expect(trace.output).toBeUndefined();
  });

  it("sends a failed trace when the stream is aborted", async () => {
    const controller = new AbortController();
    const trace = await runStream(
      async ({ abortSignal }) => ({
        stream: new ReadableStream({
          start(stream) {
            stream.enqueue(start);
            stream.enqueue({ type: "text-start", id: "1" });
            stream.enqueue({ type: "text-delta", id: "1", delta: "partial" });
            abortSignal?.addEventListener("abort", () =>
              stream.error(
                Object.assign(new Error("aborted"), { name: "AbortError" }),
              ),
            );
            setTimeout(() => controller.abort(), 0);
          },
        }),
      }),
      controller.signal,
    );
    expect(trace.error).toContain("aborted");
  });

  it("fails the trace when a later step throws after a tool ran", async () => {
    let calls = 0;
    const trace = await runStream(async () => {
      calls += 1;
      if (calls > 1) throw failure;
      return {
        stream: convertArrayToReadableStream([
          start,
          {
            type: "tool-call",
            toolCallId: "c1",
            toolName: "lookup",
            input: '{"q":"x"}',
          },
          finish("tool-calls"),
        ]),
      };
    });
    // Older v7 releases end this run with no error event, so the message varies.
    expect(trace.error).toEqual(expect.any(String));
    expect(trace.output).toBeUndefined();
    expect(trace.spans.some((s: { type: string }) => s.type === "tool")).toBe(
      true,
    );
    expect(generations(trace)).toHaveLength(2);
  });
});
