// Document passages stay in this worker. Only library/model assets are fetched.
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";
env.allowLocalModels = false;
let extractor;
self.onmessage = async ({data}) => {
  const {id, type, texts} = data;
  try {
    if (!extractor) extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
      progress_callback: p => { if (p.status === "progress" && p.total) self.postMessage({id, progress: `Downloading search model ${Math.round(p.loaded / p.total * 100)}%`}); }
    });
    if (type === "init") return self.postMessage({id, vecs: []});
    if (!Array.isArray(texts) || texts.length > 16) throw new Error("Invalid batch");
    const out = await extractor(texts, {pooling: "mean", normalize: true});
    const d = out.dims[1], vecs = texts.map((_, i) => out.data.slice(i*d, (i+1)*d));
    self.postMessage({id, vecs});
  } catch(e) { self.postMessage({id, error: String(e.message || e)}); }
};
