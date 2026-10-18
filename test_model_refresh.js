const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');

const html = fs.readFileSync(path.join(__dirname, 'static', 'index.html'), 'utf8');
const render = html.match(/  function renderDetectedModel\(s, name, glb\) \{[\s\S]*?(?=\n  function render\(s\))/)[0];
const stop = html.match(/  function stopPreview\(renderLive = true\) \{[\s\S]*?(?=\n  \$\('previewExit'\))/)[0];

function harness(autoOrient = false) {
  const attrs = new Map();
  const loads = [];
  const elements = {};
  const context = vm.createContext({
    URL, location: { href: 'http://localhost:5000/' }, lastDetection: '', lastSrc: '', preview: null,
    state: {}, ori: { yaw: 90 }, autoOrient, ORI_ORBIT: '0deg 80deg auto', IDLE_ORBIT: '45deg 75deg auto',
    viewer: {
      setAttribute(k, v) { attrs.set(k, v); if (k === 'src') loads.push(v); },
      removeAttribute(k) { attrs.delete(k); },
      closest() { return { classList: { remove() {} } }; }
    },
    $(id) { return elements[id] ||= { textContent: '', classList: { remove() {} } }; },
    renderCredit() {},
    render() { throw new Error('Detection refresh must not recursively render'); }
  });
  vm.runInContext(render + '\n' + stop, context);
  const update = (extra = {}) => {
    const s = {
      view: 'debug', detection_id: '1792358000000000001', confirmed_label: 'lime',
      viewer_label: 'lime', glb_url: '/static/models/lime.glb?v=123', asset_status: 'ready', ...extra
    };
    context.renderDetectedModel(s, s.viewer_label, s.glb_url);
  };
  return { context, attrs, loads, update, elements };
}

test('one reload per detection, including the same product detected again', () => {
  const h = harness();
  h.update();
  for (let i = 0; i < 60; i++) h.update();
  assert.equal(h.loads.length, 1);
  const first = new URL(h.loads[0]);
  assert.equal(first.searchParams.get('v'), '123');
  assert.equal(first.searchParams.get('_detection'), '1792358000000000001');
  h.update({ confirmed_label: null });
  assert.equal(h.loads.length, 1);
  h.update({ detection_id: '1792358000000000002' });
  assert.equal(h.loads.length, 2);
  assert.notEqual(h.loads[0], h.loads[1]);
  assert.equal(h.attrs.get('camera-orbit'), '45deg 75deg auto');
  assert.equal(h.attrs.get('camera-target'), 'auto auto auto');
  assert.equal(h.context.ori.yaw, null);
});

test('a new detection exits manual preview, but repeated frames leave it alone', () => {
  const h = harness();
  h.update();
  h.context.preview = { title: 'Manual library model' };
  h.update();
  assert.ok(h.context.preview);
  assert.equal(h.loads.length, 1);
  h.update({ detection_id: '2', confirmed_label: 'grapefruit', viewer_label: 'grapefruit',
    glb_url: '/static/models/grapefruit.glb?v=456' });
  assert.equal(h.context.preview, null);
  assert.match(h.attrs.get('src'), /grapefruit\.glb/);
  assert.match(h.elements.viewerLabel.textContent, /grapefruit/);
});

test('pending assets clear the old model then load when ready', () => {
  const h = harness();
  h.update();
  h.context.preview = { title: 'Old preview' };
  const detected = { view: 'smart', detection_id: '2', confirmed_label: 'lemon', viewer_label: 'lemon' };
  h.update({ ...detected, glb_url: '', asset_status: 'resolving' });
  assert.equal(h.context.preview, null);
  assert.equal(h.attrs.has('src'), false);
  assert.equal(h.loads.length, 1);
  h.update({ ...detected, glb_url: '/static/models/lemon.glb?v=789' });
  assert.equal(h.loads.length, 2);
  assert.match(h.attrs.get('src'), /lemon\.glb/);
});

test('same-detection asset revisions reload without resetting Auto-Orient controls', () => {
  const h = harness(true);
  h.attrs.set('orientation', '0deg 15deg 30deg');
  h.update();
  h.update({ glb_url: '/static/models/lime.glb?v=124' });
  assert.equal(h.loads.length, 2);
  assert.equal(h.attrs.get('camera-orbit'), '0deg 80deg auto');
  assert.equal(h.attrs.get('orientation'), '0deg 15deg 30deg');
  assert.equal(h.attrs.has('camera-controls'), false);
});
