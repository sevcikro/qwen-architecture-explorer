import buildInfo from '../generated/build-info.json';
import { Evidence } from '../components/ui';
import { EnglishOnlyNotice } from '../i18n/i18n';

export default function About() {
  return (
    <div className="page">
      <h1>About Qwen Atlas</h1>
      <EnglishOnlyNotice />
      <div className="callout">Qwen Atlas is an <strong>independent</strong> educational project. It is not affiliated with, sponsored by or endorsed by Alibaba Cloud or the Qwen team. “Qwen” is used only to identify the models described. Model weights are not redistributed, apart from seven tiny tiles (16 × 16 values or fewer) of the Apache-2.0-licensed Qwen3-8B and Qwen3-30B-A3B checkpoints, shown with attribution and provenance.</div>
      <h2>What runs in your browser</h2>
      <ul style={{ paddingLeft: 18 }}>
        <li>A tiny, seeded toy simulator with the operations and tensor names of Qwen3 (d = 8, 2 blocks). Every lab computes real numbers with it. <Evidence kind="simulation" /></li>
        <li>Structural data (shapes, names, counts, byte offsets) for Qwen3-8B and Qwen3-30B-A3B from their pinned configs and checkpoint headers. <Evidence kind="verified" /></li>
        <li>No real Qwen model is executed, and no request is sent to any AI service.</li>
      </ul>
      <h2>Known limitations</h2>
      <ul style={{ paddingLeft: 18 }}>
        <li>Actual weight values are shown only inside the shipped tiles; everywhere else the site says “value not loaded”.</li>
        <li>No real activations, attention maps or routing decisions of Qwen models are shown. All such views use the toy simulator and are labeled.</li>
        <li>The training lab is a small attention-free block trained with SGD; it illustrates the mechanics, not Qwen’s recipe.</li>
        <li>Newer hybrid models (Qwen3-Next, Qwen3.5, Qwen3.8-Flash-Next) are covered as verified structural views; Qwen3.8’s Gated DeltaNet recurrence, QSA selection and gated residual are additionally shown as toy analogues.</li>
        <li>Memory figures are theoretical estimates with formulas, not measurements of any runtime.</li>
      </ul>
      <h2>Build</h2>
      <dl className="kv">
        <dt>Source commit</dt><dd className="mono" data-testid="build-commit">{buildInfo.commit}{buildInfo.dirty ? ' (built with uncommitted changes)' : ''}</dd>
        <dt>Built at</dt><dd className="mono">{buildInfo.builtAt}</dd>
      </dl>
      <h2>Your data</h2>
      <p>The site stores only your color-theme choice in this browser’s local storage. It sets no cookies and loads no third-party scripts, fonts or analytics.</p>
    </div>
  );
}
