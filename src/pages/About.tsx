import buildInfo from '../generated/build-info.json';
import { Evidence } from '../components/ui';
import { useLang } from '../i18n/i18n';
import { useSamples } from '../data/data';

export default function About() {
  const { lang } = useLang();
  const sk = lang === 'sk';
  const samples = useSamples();
  const nTiles = samples.status === 'ok' ? samples.data.samples.length : null;
  const tiles = nTiles === null ? (sk ? 'niekoľkých' : 'a few') : String(nTiles);
  if (sk) {
    return (
      <div className="page">
        <h1>O projekte Qwen Atlas</h1>
        <div className="callout">Qwen Atlas je <strong>nezávislý</strong> vzdelávací projekt. Nie je prepojený s Alibaba Cloud ani tímom Qwen, nie je nimi sponzorovaný ani schválený. „Qwen“ sa používa len na označenie opisovaných modelov. Váhy modelov sa nešíria, okrem {tiles} malých dlaždíc (najviac 16 × 16 hodnôt) z checkpointov Qwen3-8B a Qwen3-30B-A3B s licenciou Apache-2.0, zobrazených s uvedením pôvodu.</div>
        <h2>Čo beží vo vašom prehliadači</h2>
        <ul style={{ paddingLeft: 18 }}>
          <li>Malý hračkový simulátor s osivom, s operáciami a názvami tenzorov ako Qwen3 (d = 8, 2 bloky). Každé laboratórium s ním počíta skutočné čísla. <Evidence kind="simulation" /></li>
          <li>Štrukturálne údaje (tvary, názvy, počty, bajtové posuny) pre Qwen3-8B a Qwen3-30B-A3B z ich pripnutých konfigurácií a hlavičiek checkpointov. <Evidence kind="verified" /></li>
          <li>Vopred zaznamenané aktivácie menšieho modelu Qwen3-0.6B pre tri pevné zadania (stránka „Skutočný beh“). <Evidence kind="sample" /></li>
          <li>Vo vašom prehliadači sa nespúšťa žiadny skutočný model Qwen a neodosiela sa žiadna požiadavka na službu umelej inteligencie.</li>
        </ul>
        <h2>Známe obmedzenia</h2>
        <ul style={{ paddingLeft: 18 }}>
          <li>Skutočné hodnoty váh sa zobrazujú len v dodaných dlaždiciach; všade inde stránka uvádza „hodnota nenačítaná“.</li>
          <li>Skutočné aktivácie a mapy pozornosti pochádzajú len z Qwen3-0.6B a len pre tri zadania. Pre Qwen3-8B, Qwen3-30B-A3B ani hybridné modely sa nezobrazujú žiadne skutočné aktivácie ani rozhodnutia smerovania; tieto pohľady používajú hračkový simulátor a sú tak označené.</li>
          <li>Laboratórium trénovania je malý blok bez pozornosti trénovaný cez SGD; ukazuje mechaniku, nie postup trénovania Qwen.</li>
          <li>Novšie hybridné modely (Qwen3-Next, Qwen3.5, Qwen3.8-Flash-Next) sú pokryté ako overené štrukturálne pohľady; pre Qwen3.8 sú navyše rekurencia Gated DeltaNet, výber QSA a hradlované rezíduum ukázané ako hračkové obdoby.</li>
          <li>Údaje o pamäti sú teoretické odhady so vzorcami, nie merania konkrétneho prostredia behu.</li>
          <li>Ukážky kódu v Pythone v kurze slúžia na učenie; tie, ktoré potrebujú knižnice alebo stiahnutie modelu, sú tak označené.</li>
        </ul>
        <h2>Zostavenie</h2>
        <dl className="kv">
          <dt>Zdrojový commit</dt><dd className="mono" data-testid="build-commit">{buildInfo.commit}{buildInfo.dirty ? ' (zostavené s necommitnutými zmenami)' : ''}</dd>
          <dt>Zostavené</dt><dd className="mono">{buildInfo.builtAt}</dd>
        </dl>
        <h2>Vaše údaje</h2>
        <p>Stránka si v lokálnom úložisku tohto prehliadača pamätá len zvolenú farebnú tému a jazyk. Nenastavuje žiadne cookies a nenačítava žiadne skripty, písma ani analytiku tretích strán.</p>
      </div>
    );
  }
  return (
    <div className="page">
      <h1>About Qwen Atlas</h1>
      <div className="callout">Qwen Atlas is an <strong>independent</strong> educational project. It is not affiliated with, sponsored by or endorsed by Alibaba Cloud or the Qwen team. “Qwen” is used only to identify the models described. Model weights are not redistributed, apart from {tiles} tiny tiles (16 × 16 values or fewer) of the Apache-2.0-licensed Qwen3-8B and Qwen3-30B-A3B checkpoints, shown with attribution and provenance.</div>
      <h2>What runs in your browser</h2>
      <ul style={{ paddingLeft: 18 }}>
        <li>A tiny, seeded toy simulator with the operations and tensor names of Qwen3 (d = 8, 2 blocks). Every lab computes real numbers with it. <Evidence kind="simulation" /></li>
        <li>Structural data (shapes, names, counts, byte offsets) for Qwen3-8B and Qwen3-30B-A3B from their pinned configs and checkpoint headers. <Evidence kind="verified" /></li>
        <li>Pre-recorded activations of the smaller Qwen3-0.6B for three fixed prompts (the “Real run” page). <Evidence kind="sample" /></li>
        <li>No real Qwen model is executed in your browser, and no request is sent to any AI service.</li>
      </ul>
      <h2>Known limitations</h2>
      <ul style={{ paddingLeft: 18 }}>
        <li>Actual weight values are shown only inside the shipped tiles; everywhere else the site says “value not loaded”.</li>
        <li>Real activations and attention maps come only from Qwen3-0.6B and only for three prompts. No real activations or routing decisions of Qwen3-8B, Qwen3-30B-A3B or the hybrid models are shown; those views use the toy simulator and are labeled.</li>
        <li>The training lab is a small attention-free block trained with SGD; it illustrates the mechanics, not Qwen’s recipe.</li>
        <li>Newer hybrid models (Qwen3-Next, Qwen3.5, Qwen3.8-Flash-Next) are covered as verified structural views; Qwen3.8’s Gated DeltaNet recurrence, QSA selection and gated residual are additionally shown as toy analogues.</li>
        <li>Memory figures are theoretical estimates with formulas, not measurements of any runtime.</li>
        <li>Python code in the course is for learning; snippets that need libraries or a model download say so.</li>
      </ul>
      <h2>Build</h2>
      <dl className="kv">
        <dt>Source commit</dt><dd className="mono" data-testid="build-commit">{buildInfo.commit}{buildInfo.dirty ? ' (built with uncommitted changes)' : ''}</dd>
        <dt>Built at</dt><dd className="mono">{buildInfo.builtAt}</dd>
      </dl>
      <h2>Your data</h2>
      <p>The site stores only your color-theme and language choice in this browser’s local storage. It sets no cookies and loads no third-party scripts, fonts or analytics.</p>
    </div>
  );
}
