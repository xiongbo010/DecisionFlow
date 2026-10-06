const demoRoot = document.querySelector('.demo-shell');

if (demoRoot) {
  const inputs = {
    route: document.querySelector('#route-score'),
    fraud: document.querySelector('#fraud-score'),
    urgency: document.querySelector('#urgency-score'),
    action: document.querySelector('#action-score'),
  };

  const presets = {
    conflicted: { route: 42, fraud: 78, urgency: 64, action: 54 },
    uncertain: { route: 49, fraud: 51, urgency: 40, action: 36 },
    aligned: { route: 74, fraud: 71, urgency: 58, action: 20 },
  };

  const policyLabels = {
    fraudRoute: 'Fraud requires Security',
    fraudAction: 'Fraud excludes Refund only',
    highAction: 'High urgency excludes Refund only',
    securityAction: 'Security excludes Refund only',
  };

  const labels = {
    route: { billing: 'Billing', security: 'Security' },
    fraud: { no: 'No fraud', yes: 'Fraud' },
    urgency: { low: 'Low', medium: 'Medium', high: 'High' },
    action: { refund: 'Refund only', secure: 'Secure account', review: 'Manual review' },
  };

  const percent = (value, digits = 0) => `${(value * 100).toFixed(digits)}%`;

  const distributions = () => {
    const security = Number(inputs.route.value) / 100;
    const fraud = Number(inputs.fraud.value) / 100;
    const high = Number(inputs.urgency.value) / 100;
    const refund = Number(inputs.action.value) / 100;
    return {
      route: { billing: 1 - security, security },
      fraud: { no: 1 - fraud, yes: fraud },
      urgency: { low: (1 - high) * 0.3, medium: (1 - high) * 0.7, high },
      action: { refund, secure: (1 - refund) * 0.65, review: (1 - refund) * 0.35 },
    };
  };

  const activePolicies = () => Object.fromEntries(
    [...document.querySelectorAll('[data-policy]')].map((input) => [input.dataset.policy, input.checked])
  );

  const violationsFor = (world, policies) => {
    const violations = [];
    if (policies.fraudRoute && world.fraud === 'yes' && world.route !== 'security') violations.push('fraudRoute');
    if (policies.fraudAction && world.fraud === 'yes' && world.action === 'refund') violations.push('fraudAction');
    if (policies.highAction && world.urgency === 'high' && world.action === 'refund') violations.push('highAction');
    if (policies.securityAction && world.route === 'security' && world.action === 'refund') violations.push('securityAction');
    return violations;
  };

  const enumerateWorlds = (probabilities, policies) => {
    const worlds = [];
    Object.keys(probabilities.route).forEach((route) => {
      Object.keys(probabilities.fraud).forEach((fraud) => {
        Object.keys(probabilities.urgency).forEach((urgency) => {
          Object.keys(probabilities.action).forEach((action) => {
            const world = { route, fraud, urgency, action };
            const weight = probabilities.route[route]
              * probabilities.fraud[fraud]
              * probabilities.urgency[urgency]
              * probabilities.action[action];
            const violations = violationsFor(world, policies);
            worlds.push({ ...world, weight, violations, valid: violations.length === 0 });
          });
        });
      });
    });
    const validWorlds = worlds.filter((world) => world.valid);
    const z = validWorlds.reduce((sum, world) => sum + world.weight, 0);
    validWorlds.forEach((world) => { world.posterior = z > 0 ? world.weight / z : 0; });
    validWorlds.sort((a, b) => b.posterior - a.posterior);
    return { worlds, validWorlds, z };
  };

  const argmax = (distribution) => Object.entries(distribution).sort((a, b) => b[1] - a[1])[0][0];

  const localWorld = (probabilities, policies) => {
    const world = {
      route: argmax(probabilities.route),
      fraud: argmax(probabilities.fraud),
      urgency: argmax(probabilities.urgency),
      action: argmax(probabilities.action),
    };
    return { ...world, violations: violationsFor(world, policies) };
  };

  const marginalsFor = (validWorlds) => {
    const marginals = {
      route: { billing: 0, security: 0 },
      fraud: { no: 0, yes: 0 },
      urgency: { low: 0, medium: 0, high: 0 },
      action: { refund: 0, secure: 0, review: 0 },
    };
    validWorlds.forEach((world) => {
      Object.keys(marginals).forEach((variable) => {
        marginals[variable][world[variable]] += world.posterior;
      });
    });
    return marginals;
  };

  const pathMarkup = (world) => [
    ['Route', labels.route[world.route]],
    ['Fraud', labels.fraud[world.fraud]],
    ['Urgency', labels.urgency[world.urgency]],
    ['Action', labels.action[world.action]],
  ].map(([name, value]) => `<div class="path-step"><span>${name}</span><strong>${value}</strong></div>`).join('');

  const worldLabel = (world) => `${labels.route[world.route]} · ${labels.fraud[world.fraud]} · ${labels.urgency[world.urgency]} · ${labels.action[world.action]}`;

  const updateScoreLabels = (probabilities) => {
    const updates = {
      'route-value': probabilities.route.security,
      'route-billing': probabilities.route.billing,
      'route-security': probabilities.route.security,
      'fraud-value': probabilities.fraud.yes,
      'fraud-no': probabilities.fraud.no,
      'fraud-yes': probabilities.fraud.yes,
      'urgency-value': probabilities.urgency.high,
      'urgency-rest': 1 - probabilities.urgency.high,
      'urgency-high': probabilities.urgency.high,
      'action-value': probabilities.action.refund,
      'action-rest': 1 - probabilities.action.refund,
      'action-refund': probabilities.action.refund,
    };
    Object.entries(updates).forEach(([id, value]) => { document.querySelector(`#${id}`).textContent = percent(value); });
  };

  const renderTopWorlds = (validWorlds) => {
    const list = document.querySelector('#top-world-list');
    list.innerHTML = validWorlds.slice(0, 4).map((world, index) => `
      <li>
        <span class="world-rank">${String(index + 1).padStart(2, '0')}</span>
        <span class="world-label">${worldLabel(world)}</span>
        <strong class="world-probability">${percent(world.posterior, 1)}</strong>
      </li>
    `).join('');
  };

  const renderMarginals = (marginals) => {
    const titles = { route: 'Route marginal', fraud: 'Fraud marginal', urgency: 'Urgency marginal', action: 'Action marginal' };
    document.querySelector('#marginal-grid').innerHTML = Object.entries(marginals).map(([variable, values]) => `
      <article class="marginal-card">
        <strong>${titles[variable]}</strong>
        ${Object.entries(values).map(([value, probability]) => `
          <div class="marginal-row">
            <div><span>${labels[variable][value]}</span><b>${percent(probability, 1)}</b></div>
            <span class="marginal-track"><i style="--marginal:${percent(probability, 1)}"></i></span>
          </div>
        `).join('')}
      </article>
    `).join('');
  };

  const render = () => {
    const probabilities = distributions();
    const policies = activePolicies();
    const inference = enumerateWorlds(probabilities, policies);
    const local = localWorld(probabilities, policies);
    const map = inference.validWorlds[0];
    updateScoreLabels(probabilities);

    document.querySelector('#valid-world-count').textContent = inference.validWorlds.length;
    document.querySelector('#valid-mass').textContent = inference.z.toFixed(3);
    document.querySelector('#worlds-removed').textContent = percent((inference.worlds.length - inference.validWorlds.length) / inference.worlds.length);
    document.querySelector('#local-path').innerHTML = pathMarkup(local);
    document.querySelector('#map-path').innerHTML = pathMarkup(map);

    const localStatus = document.querySelector('#local-status');
    localStatus.textContent = local.violations.length ? `${local.violations.length} conflict${local.violations.length === 1 ? '' : 's'}` : 'Coherent';
    localStatus.classList.toggle('is-coherent', local.violations.length === 0);

    const violationList = document.querySelector('#local-violations');
    violationList.innerHTML = local.violations.length
      ? local.violations.map((violation) => `<li>${policyLabels[violation]}</li>`).join('')
      : '<li class="no-conflict">Local output already satisfies every active policy</li>';

    document.querySelector('#map-probability').innerHTML = `Posterior probability <b>${percent(map.posterior, 1)}</b> after normalizing the valid mass.`;
    renderTopWorlds(inference.validWorlds);
    renderMarginals(marginalsFor(inference.validWorlds));
  };

  Object.values(inputs).forEach((input) => {
    input.addEventListener('input', () => {
      document.querySelectorAll('.preset-button').forEach((button) => button.classList.remove('active'));
      render();
    });
  });

  document.querySelectorAll('[data-policy]').forEach((input) => input.addEventListener('change', render));

  document.querySelectorAll('[data-preset]').forEach((button) => {
    button.addEventListener('click', () => {
      const preset = presets[button.dataset.preset];
      Object.entries(preset).forEach(([name, value]) => { inputs[name].value = value; });
      document.querySelectorAll('.preset-button').forEach((candidate) => candidate.classList.toggle('active', candidate === button));
      render();
    });
  });

  render();
}
