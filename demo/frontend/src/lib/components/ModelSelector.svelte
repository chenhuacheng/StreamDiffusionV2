<script lang="ts">
  import { onMount } from 'svelte';

  type Variant = { id: string; label?: string; port: number };
  let current: string = '';
  let variants: Variant[] = [];
  let loaded = false;

  onMount(async () => {
    try {
      const res = await fetch('/api/variant');
      if (!res.ok) return;
      const data = await res.json();
      current = data.current || '';
      variants = Array.isArray(data.variants) ? data.variants : [];
    } catch (err) {
      // silently ignore - backend may not support /api/variant yet
      console.warn('ModelSelector: /api/variant not available', err);
    } finally {
      loaded = true;
    }
  });

  function switchTo(v: Variant) {
    if (!v || !v.port) return;
    if (v.id && v.id === current) return;
    // Navigate to sibling backend on same host. We rely on the peer being
    // reachable on the same hostname as the current page, only the port
    // differs. This keeps the implementation trivial and the WebSocket
    // URL construction in lcmLive.ts (which uses window.location.host)
    // automatically points at the new backend after the reload.
    const url = new URL(window.location.href);
    url.port = String(v.port);
    window.location.href = url.toString();
  }
</script>

{#if loaded && variants.length > 1}
  <div class="my-2 flex flex-row items-center justify-center gap-2 text-sm">
    <span class="font-semibold text-gray-700 dark:text-gray-200">Model:</span>
    {#each variants as v (v.id)}
      <button
        type="button"
        on:click={() => switchTo(v)}
        class="rounded px-3 py-1 border transition-colors
          {v.id === current
            ? 'bg-blue-600 text-white border-blue-600 cursor-default'
            : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-100 dark:bg-gray-800 dark:text-gray-200 dark:border-gray-600 dark:hover:bg-gray-700'}"
        disabled={v.id === current}
        aria-pressed={v.id === current}
        title={v.id === current ? 'Current model' : `Switch to ${v.label || v.id}`}
      >
        {v.label || v.id}
      </button>
    {/each}
  </div>
{/if}
