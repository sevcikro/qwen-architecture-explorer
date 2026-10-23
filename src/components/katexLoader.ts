// Loaded on demand the first time a formula is shown, so KaTeX (script, styles, fonts) stays out of the main bundle.
import 'katex/dist/katex.min.css';
import katex from 'katex';
export default katex;
