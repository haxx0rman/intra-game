import { ModelType } from "@/components/modelselector";
import { openrouterCode } from "@/components/openrouter";
import { signal } from "@preact/signals-react";
import OpenAI from "openai";
import { persistentSignal } from "./persistentsignal";
import { ChatType, LlmLogType } from "./types";

export const DEFAULT_PRO_MODEL = "gemma3:27b";
export const DEFAULT_FLASH_MODEL = "gpt-oss:20b";
export const DEFAULT_MODEL = DEFAULT_PRO_MODEL;

// Ollama configuration
export const OLLAMA_BASE_URL = "http://brainmachine:11434/v1";
export const OLLAMA_DEFAULT_MODEL = "gemma3:27b";

export const customEndpoint = persistentSignal<string | null>(
  "customEndpoint",
  OLLAMA_BASE_URL
);
export const openrouterModel = persistentSignal<ModelType | null>(
  "openrouter",
  null
);

export const logSignal = signal<LlmLogType[]>([]);

export const lastLlmError = signal<string | null>(null);
export const lastLlmErrorType = signal<"openrouter" | undefined>();

export class OpenRouterError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "OpenRouterError";
  }
}

// Utility function to check if endpoint is Ollama
export function isOllamaEndpoint(endpoint: string | null): boolean {
  if (!endpoint) return false;
  return endpoint.includes("11434") || 
         endpoint.toLowerCase().includes("ollama") ||
         endpoint === OLLAMA_BASE_URL;
}

// Utility function to test Ollama connection
export async function testOllamaConnection(endpoint: string = OLLAMA_BASE_URL): Promise<boolean> {
  try {
    const response = await fetch(endpoint.replace('/v1', '/api/tags'), {
      method: 'GET',
    });
    return response.ok;
  } catch {
    return false;
  }
}

export async function chat(request: ChatType) {
  request = upliftInstructions(request);
  const log = {
    request,
  };
  const lastIndex = logSignal.value.length
    ? logSignal.value[0].request.meta.index
    : 0;
  request.meta.index = (lastIndex || 0) + 1;
  request.meta.start = Date.now();
  let model: string = DEFAULT_MODEL;
  if (!request.model) {
    model = DEFAULT_MODEL;
  } else if (request.model === "pro") {
    model = DEFAULT_PRO_MODEL;
  } else if (request.model === "flash") {
    model = DEFAULT_FLASH_MODEL;
  }
  logSignal.value = [log, ...logSignal.value.slice(0, 20)];
  let text = "";
  try {
    let openai: OpenAI;

    if (customEndpoint.value) {
      // Use custom endpoint (Ollama or other OpenAI-compatible API)
      const usingOllama = isOllamaEndpoint(customEndpoint.value);
      
      openai = new OpenAI({
        baseURL: customEndpoint.value,
        apiKey: usingOllama ? "dummy" : "dummy", // Ollama doesn't require a real API key
        dangerouslyAllowBrowser: true,
      });
      
      // Use Ollama default model if using Ollama endpoint and no specific model selected
      if (usingOllama && !openrouterModel.value) {
        model = OLLAMA_DEFAULT_MODEL;
      }
    } else {
      if (!openrouterCode.value) {
        throw new OpenRouterError(
          "No OpenRouter API key found. Please connect to OpenRouter first."
        );
      }
      if (!openrouterModel.value) {
        throw new OpenRouterError(
          "No OpenRouter model selected. Please select a model first."
        );
      }

      openai = new OpenAI({
        baseURL: "http://brainmachine:11434/v1",
        apiKey: openrouterCode.value,
        defaultHeaders: {
          "X-Title": "Intra",
          "HTTP-Referer": location.origin,
        },
        dangerouslyAllowBrowser: true,
      });
    }

    const messages = request.messages;

    const completion = await openai.chat.completions.create({
      model: openrouterModel.value?.id || model,
      messages,
      max_tokens: 30000,
    });

    if (!completion.choices[0]?.message?.content) {
      console.error("Bad Response", completion);
      lastLlmError.value = `Bad response from LLM: no content in choices`;
      throw new Error("Bad response from LLM: no content in choices");
    }

    text = completion.choices[0].message.content;
  } catch (e) {
    const newLog = {
      ...log,
      end: Date.now(),
      errorMessage: `${e}`,
    };
    logSignal.value = logSignal.value.map((l) => (l === log ? newLog : l));
    lastLlmError.value = `Unexpected LLM error: ${e}`;
    if (e instanceof OpenRouterError) {
      lastLlmErrorType.value = "openrouter";
    } else {
      lastLlmErrorType.value = undefined;
    }
    throw e;
  }
  const newLog = {
    ...log,
    end: Date.now(),
    response: text,
  };
  logSignal.value = logSignal.value.map((l) => (l === log ? newLog : l));
  return text as string;
}

function upliftInstructions(chat: ChatType): ChatType {
  const newChat = { ...chat };
  const allInstructions: string[] = [];

  // Process system messages for instructions
  newChat.messages = newChat.messages.map((message) => {
    if (message.role === "system") {
      const { repl, instructions } = parseInstructions(message.content);
      allInstructions.push(...instructions);
      return { ...message, content: repl };
    }
    return message;
  });

  // Process user and assistant messages for instructions
  newChat.messages = newChat.messages.map((message) => {
    if (message.role === "user" || message.role === "assistant") {
      const { repl, instructions } = parseInstructions(message.content);
      allInstructions.push(...instructions);
      return { ...message, content: repl };
    }
    return message;
  });

  // If we found instructions, insert them into the first system message or create one
  if (allInstructions.length > 0) {
    const systemMessages = newChat.messages.filter(
      (msg) => msg.role === "system"
    );
    if (systemMessages.length > 0) {
      // Insert into the first system message
      const firstSystemIndex = newChat.messages.findIndex(
        (msg) => msg.role === "system"
      );
      const firstSystem = newChat.messages[firstSystemIndex];
      if (firstSystem.content.includes("<insert-system />")) {
        newChat.messages[firstSystemIndex] = {
          ...firstSystem,
          content: firstSystem.content.replace(
            /<insert-system\s*\/>/i,
            allInstructions.join("\n")
          ),
        };
      } else {
        throw new Error(
          "Instructions were not inserted into system instruction"
        );
      }
    } else {
      // Create a new system message at the beginning
      newChat.messages.unshift({
        role: "system",
        content: allInstructions.join("\n"),
      });
    }
  }

  return newChat;
}

function parseInstructions(system: string): {
  repl: string;
  instructions: string[];
} {
  const instructions: string[] = [];
  const instructionRegex = /<system>([^]*?)<\/system>\s*/gi;
  const repl = system.replace(instructionRegex, (match, contents) => {
    instructions.push(contents.trim());
    return "";
  });
  return { repl, instructions };
}
