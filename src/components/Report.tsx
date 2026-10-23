import { useLang } from '../i18n/i18n';
import buildInfo from '../generated/build-info.json';

// "Report a mistake": opens a pre-filled GitHub issue. The link carries only the page address, the language and the build id.
export function ReportLink({ className = '' }: { className?: string }) {
  const { lang } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const page = window.location.hash || '#/';
  const title = `Mistake on ${page.split('?')[0]}`;
  const body = [
    `Page: ${page}`, `Language: ${lang}`, `Build: ${buildInfo.shortCommit}`, '',
    'What is wrong:', '', '', 'What would be correct (with a source if possible):', '', '',
  ].join('\n');
  const href = `https://github.com/sevcikro/qwen-architecture-explorer/issues/new?title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}&labels=content`;
  return <a className={className} href={href} target="_blank" rel="noreferrer" title={T('Opens a pre-filled issue on GitHub (free account needed)', 'Otvorí vopred vyplnené hlásenie na GitHube (treba bezplatný účet)')}>{T('Report a mistake', 'Nahlásiť chybu')}</a>;
}
