import { useState } from "react";
import { TranscriptUtterance } from "../types/meeting";

export function useTranscript() {
  const [lines, setLines] = useState<TranscriptUtterance[]>([]);

  const appendUtterances = (utterances: TranscriptUtterance[]) => {
    setLines((current) => {
      const merged = new Map(current.map((u) => [u.id, u]));
      utterances.forEach((u) => merged.set(u.id, u));
      return Array.from(merged.values()).sort((a, b) => a.start_time - b.start_time);
    });
  };

  const applySpeakerBackfill = (updates: Array<{ utterance_id: string; speaker: string }>) => {
    if (updates.length === 0) return;
    const speakerMap = new Map(updates.map((u) => [u.utterance_id, u.speaker]));
    setLines((current) =>
      current.map((line) =>
        speakerMap.has(line.id)
          ? { ...line, speaker: speakerMap.get(line.id) ?? line.speaker }
          : line
      )
    );
  };

  const editLine = (id: string, text: string) => {
    setLines((current) =>
      current.map((line) => (line.id === id ? { ...line, text } : line))
    );
  };

  const reset = () => setLines([]);

  return { lines, appendUtterances, applySpeakerBackfill, editLine, reset };
}
