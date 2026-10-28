import { mount } from "svelte";
import App from "./App.svelte";

// Send the browser back to the login page when its session lapses.
const nativeFetch = window.fetch.bind(window);
window.fetch = async (...args) => {
  const response = await nativeFetch(...args);
  const loginPath = response.headers.get("x-modal-dojo-login");
  if (response.status === 401 && loginPath) {
    const next = window.location.pathname + window.location.search;
    window.location.assign(`${loginPath}?next=${encodeURIComponent(next)}`);
  }
  return response;
};

const app = mount(App, {
  target: document.getElementById("app"),
});

export default app;
