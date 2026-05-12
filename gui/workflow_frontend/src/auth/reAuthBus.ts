export type ReAuthReason = "refresh-failed" | "api-401";

type Listener = (reason: ReAuthReason) => void;

let listeners: Listener[] = [];

export const reAuthBus = {
  emit(reason: ReAuthReason) {
    listeners.forEach((fn) => {
      try {
        fn(reason);
      } catch (err) {
        console.error("reAuthBus listener error:", err);
      }
    });
  },
  subscribe(fn: Listener): () => void {
    listeners.push(fn);
    return () => {
      listeners = listeners.filter((l) => l !== fn);
    };
  },
};
