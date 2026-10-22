(function () {
  function slug(t) { return t.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 48); }
  // LOG_SECTIONS and LOG_EPAGES (written by build.py): every page's sections, and the page each E-code's anchor is on.
  // In the nav, this page's sections expand and the others link to their page.
  var sections = Array.prototype.slice.call(document.querySelectorAll('main section[id]'));
  var outline = LOG_SECTIONS.map(function (s) {
    var sec = document.getElementById(s.id);
    if (!sec) return { id: s.id, href: s.page + '#' + s.id, num: s.num, label: s.label, el: null, subs: [] };
    var subs = Array.prototype.slice.call(sec.querySelectorAll('h3')).filter(function (h3) { return !h3.closest('.card') || h3.closest('.exp'); }).map(function (h3) {
      var host = h3.closest('.exp[id]');
      var id = host ? host.id : (h3.id || (sec.id + '-' + slug(h3.textContent)));
      if (!host) h3.id = id;
      var code = host ? host.querySelector('.top .id') : null;
      return { id: id, code: code ? code.textContent.trim() : '', label: h3.textContent.trim(), el: host || h3 };
    });
    return { id: s.id, href: '#' + s.id, num: s.num, label: s.label, el: sec, subs: subs };
  });
  var local = outline.filter(function (s) { return s.el; });
  function render(listEl, withNums) {
    outline.forEach(function (s) {
      var li = document.createElement('li'), a = document.createElement('a');
      a.href = s.href;
      if (withNums) { var n = document.createElement('span'); n.className = 'num'; n.textContent = s.num; a.appendChild(n); }
      a.appendChild(document.createTextNode(s.label)); li.appendChild(a);
      if (s.subs.length) {
        var ol = document.createElement('ol');
        s.subs.forEach(function (sub) { var sli = document.createElement('li'), sa = document.createElement('a'); sa.href = '#' + sub.id; if (sub.code) { var c = document.createElement('span'); c.className = 'num'; c.textContent = sub.code; sa.appendChild(c); } sa.appendChild(document.createTextNode(sub.label)); sli.appendChild(sa); ol.appendChild(sli); });
        li.appendChild(ol);
      }
      listEl.appendChild(li);
    });
  }
  render(document.getElementById('toc-list'), true);
  render(document.getElementById('mobile-toc-list'), false);

  var toc = document.getElementById('toc-list');
  var topLis = Array.prototype.slice.call(toc.children);
  var topLinks = topLis.map(function (li) { return li.querySelector(':scope > a'); });
  var subLinks = {};
  Array.prototype.slice.call(toc.querySelectorAll('ol a')).forEach(function (a) { subLinks[a.getAttribute('href').slice(1)] = a; });
  var cur = { sec: null, sub: null };
  function update() {
    var y = window.scrollY + 120, s = local[0], sub = null;
    local.forEach(function (l) { if (l.el.offsetTop <= y) s = l; });
    for (var j = 0; j < s.subs.length; j++) if (s.subs[j].el.offsetTop <= y) sub = s.subs[j].id;
    if (s.id === cur.sec && sub === cur.sub) return;
    cur = { sec: s.id, sub: sub };
    topLis.forEach(function (li, k) { li.classList.toggle('open', outline[k] === s); topLinks[k].classList.toggle('active', outline[k] === s && !sub); });
    Object.keys(subLinks).forEach(function (id) { subLinks[id].classList.toggle('active', id === sub); });
    var act = toc.querySelector('a.active');
    if (act) { var nav = toc.parentNode, r = act.getBoundingClientRect(), nr = nav.getBoundingClientRect(); if (r.top < nr.top + 40 || r.bottom > nr.bottom - 40) act.scrollIntoView({ block: 'center' }); }
  }
  window.addEventListener('scroll', update, { passive: true });
  window.addEventListener('resize', update);
  update();

  // Link E-code references in tables, cards and field lists.
  var scopes = document.querySelectorAll('#claims td:last-child, #apparatus td:last-child, .exp dd, .card p, .card li, .q .status, .terms p');
  Array.prototype.slice.call(scopes).forEach(function (cell) {
    var walker = document.createTreeWalker(cell, NodeFilter.SHOW_TEXT), nodes = [], n;
    while ((n = walker.nextNode())) if (/\bE\d{1,2}\b/.test(n.nodeValue) && !n.parentNode.closest('a')) nodes.push(n);
    nodes.forEach(function (tn) {
      var frag = document.createDocumentFragment(), re = /\bE(\d{1,2})\b/g, last = 0, m, txt = tn.nodeValue;
      while ((m = re.exec(txt))) {
        frag.appendChild(document.createTextNode(txt.slice(last, m.index)));
        var id = 'e' + m[1], a = document.createElement('a'); a.href = (document.getElementById(id) ? '' : LOG_EPAGES[id] || '') + '#' + id; a.textContent = m[0]; frag.appendChild(a);
        last = m.index + m[0].length;
      }
      frag.appendChild(document.createTextNode(txt.slice(last)));
      tn.parentNode.replaceChild(frag, tn);
    });
  });

  sections.forEach(function (sec) { var a = document.createElement('a'); a.className = 'up'; a.href = '#mobile-toc'; a.textContent = '↑ Contents'; sec.appendChild(a); });
  var mt = document.getElementById('mobile-toc');
  mt.addEventListener('click', function (e) { if (e.target.tagName === 'A') mt.removeAttribute('open'); });
  // A link opened in a new tab arrives on the main page with only its #anchor (the host drops the page name), so an
  // anchor held by another page is forwarded there. Scroll again once the figures have loaded and moved the layout.
  var hash = location.hash.slice(1);
  if (hash) {
    var t = document.getElementById(hash);
    var here = (LOG_SECTIONS.filter(function (s) { return document.getElementById(s.id); })[0] || {}).page;
    var away = LOG_EPAGES[hash] || (LOG_SECTIONS.filter(function (s) { return s.id === hash; })[0] || {}).page;
    if (!t && away && away !== here) location.replace(away + '#' + hash);
    else if (t) settle(t);
  }
  // Keep the target in view while fonts and figures load and the layout grows, until the reader scrolls or 10 s pass.
  function settle(t) {
    var stop = false, go = function () { if (!stop) t.scrollIntoView(); };
    var ro = new ResizeObserver(go);
    ro.observe(document.body);
    ['wheel', 'touchstart', 'keydown', 'mousedown'].forEach(function (e) { window.addEventListener(e, function () { stop = true; ro.disconnect(); }, { once: true, passive: true }); });
    setTimeout(function () { stop = true; ro.disconnect(); }, 10000);
    go();
  }
})();
