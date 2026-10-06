const trajectoryRoot = document.querySelector('.trajectory-shell');

if (trajectoryRoot) {
  const inputs = {
    route: document.querySelector('#ms-route-score'),
    fraud: document.querySelector('#ms-fraud-score'),
    urgency: document.querySelector('#ms-urgency-score'),
    handling: document.querySelector('#ms-handling-score'),
    followup: document.querySelector('#ms-followup-score'),
  };

  const presets = {
    conflicted: { route: 42, fraud: 78, urgency: 64, handling: 68, followup: 66 },
    uncertain: { route: 49, fraud: 51, urgency: 45, handling: 48, followup: 50 },
    aligned: { route: 72, fraud: 68, urgency: 55, handling: 65, followup: 62 },
  };

  const labels = {
    route: { billing: 'Billing', security: 'Security' },
    fraud: { no: 'No fraud', yes: 'Fraud' },
    urgency: { normal: 'Normal', high: 'High' },
    action: { refund: 'Refund charge', secure: 'Secure account', review: 'Manual review' },
    followup: { close: 'Close ticket', notify: 'Notify & monitor', escalate: 'Escalate' },
  };

  const policyLabels = {
    fraudRoute: 'Fraud requires the Security route',
    securityAction: 'Security excludes Refund charge',
    billingAction: 'Billing excludes Secure account',
    fraudClosure: 'Fraud excludes Close ticket',
    highClosure: 'High urgency excludes Close ticket',
    reviewEscalate: 'Manual review must be followed by Escalate',
  };

  const percent = (value, digits = 0) => `${(value * 100).toFixed(digits)}%`;
  const argmax = (distribution) => Object.entries(distribution).sort((a, b) => b[1] - a[1])[0][0];

  const initialDistributions = () => {
    const security = Number(inputs.route.value) / 100;
    const fraud = Number(inputs.fraud.value) / 100;
    const high = Number(inputs.urgency.value) / 100;
    return {
      route: { billing: 1 - security, security },
      fraud: { no: 1 - fraud, yes: fraud },
      urgency: { normal: 1 - high, high },
    };
  };

  const actionDistribution = (route) => {
    const aligned = Number(inputs.handling.value) / 100;
    if (route === 'security') {
      return { refund: (1 - aligned) * 0.3, secure: aligned, review: (1 - aligned) * 0.7 };
    }
    return { refund: aligned, secure: (1 - aligned) * 0.3, review: (1 - aligned) * 0.7 };
  };

  const followupDistribution = (action) => {
    const aligned = Number(inputs.followup.value) / 100;
    if (action === 'secure') {
      return { close: (1 - aligned) * 0.3, notify: aligned, escalate: (1 - aligned) * 0.7 };
    }
    if (action === 'refund') {
      return { close: aligned, notify: (1 - aligned) * 0.7, escalate: (1 - aligned) * 0.3 };
    }
    return { close: (1 - aligned) * 0.3, notify: (1 - aligned) * 0.7, escalate: aligned };
  };

  const activePolicies = () => Object.fromEntries(
    [...document.querySelectorAll('[data-trajectory-policy]')].map((input) => [input.dataset.trajectoryPolicy, input.checked])
  );

  const violationsFor = (trajectory, policies) => {
    const violations = [];
    if (policies.fraudRoute && trajectory.fraud === 'yes' && trajectory.route !== 'security') violations.push('fraudRoute');
    if (policies.securityAction && trajectory.route === 'security' && trajectory.action === 'refund') violations.push('securityAction');
    if (policies.billingAction && trajectory.route === 'billing' && trajectory.action === 'secure') violations.push('billingAction');
    if (policies.fraudClosure && trajectory.fraud === 'yes' && trajectory.followup === 'close') violations.push('fraudClosure');
    if (policies.highClosure && trajectory.urgency === 'high' && trajectory.followup === 'close') violations.push('highClosure');
    if (policies.reviewEscalate && trajectory.action === 'review' && trajectory.followup !== 'escalate') violations.push('reviewEscalate');
    return violations;
  };

  const enumerateTrajectories = (initial, policies) => {
    const trajectories = [];
    Object.entries(initial.route).forEach(([route, routeProbability]) => {
      Object.entries(initial.fraud).forEach(([fraud, fraudProbability]) => {
        Object.entries(initial.urgency).forEach(([urgency, urgencyProbability]) => {
          Object.entries(actionDistribution(route)).forEach(([action, actionProbability]) => {
            Object.entries(followupDistribution(action)).forEach(([followup, followupProbability]) => {
              const trajectory = { route, fraud, urgency, action, followup };
              const weight = routeProbability * fraudProbability * urgencyProbability * actionProbability * followupProbability;
              const violations = violationsFor(trajectory, policies);
              trajectories.push({ ...trajectory, weight, violations, valid: violations.length === 0 });
            });
          });
        });
      });
    });
    const valid = trajectories.filter((trajectory) => trajectory.valid);
    const z = valid.reduce((sum, trajectory) => sum + trajectory.weight, 0);
    valid.forEach((trajectory) => { trajectory.posterior = z > 0 ? trajectory.weight / z : 0; });
    valid.sort((a, b) => b.posterior - a.posterior);
    return { trajectories, valid, z };
  };

  const greedyTrajectory = (initial, policies) => {
    const route = argmax(initial.route);
    const fraud = argmax(initial.fraud);
    const urgency = argmax(initial.urgency);
    const action = argmax(actionDistribution(route));
    const followup = argmax(followupDistribution(action));
    const trajectory = { route, fraud, urgency, action, followup };
    return { ...trajectory, violations: violationsFor(trajectory, policies) };
  };

  const marginalsFor = (valid) => {
    const marginals = {
      route: { billing: 0, security: 0 },
      action: { refund: 0, secure: 0, review: 0 },
      followup: { close: 0, notify: 0, escalate: 0 },
    };
    valid.forEach((trajectory) => {
      Object.keys(marginals).forEach((variable) => {
        marginals[variable][trajectory[variable]] += trajectory.posterior;
      });
    });
    return marginals;
  };

  const trajectoryMarkup = (trajectory) => `
    <div class="timeline-step">
      <span>Step 1 · Intake</span>
      <strong>${labels.route[trajectory.route]}</strong>
      <small>${labels.fraud[trajectory.fraud]} · ${labels.urgency[trajectory.urgency]}</small>
    </div>
    <i aria-hidden="true">→</i>
    <div class="timeline-step">
      <span>Step 2 · Resolve</span>
      <strong>${labels.action[trajectory.action]}</strong>
      <small>conditioned on ${labels.route[trajectory.route]}</small>
    </div>
    <i aria-hidden="true">→</i>
    <div class="timeline-step">
      <span>Step 3 · Follow up</span>
      <strong>${labels.followup[trajectory.followup]}</strong>
      <small>conditioned on ${labels.action[trajectory.action]}</small>
    </div>`;

  const trajectoryLabel = (trajectory) => `${labels.route[trajectory.route]} · ${labels.fraud[trajectory.fraud]} · ${labels.urgency[trajectory.urgency]} → ${labels.action[trajectory.action]} → ${labels.followup[trajectory.followup]}`;

  const renderRankedTrajectories = (valid) => {
    document.querySelector('#ms-top-trajectories').innerHTML = valid.slice(0, 5).map((trajectory, index) => `
      <li>
        <span>${String(index + 1).padStart(2, '0')}</span>
        <div><strong>${trajectoryLabel(trajectory)}</strong><small>complete valid trajectory</small></div>
        <b>${percent(trajectory.posterior, 1)}</b>
      </li>`).join('');
  };

  const renderMarginals = (marginals, initial) => {
    const groups = [
      { title: 'Step 1 · Route', variable: 'route', values: marginals.route, raw: initial.route },
      { title: 'Step 2 · Action', variable: 'action', values: marginals.action },
      { title: 'Step 3 · Outcome', variable: 'followup', values: marginals.followup },
    ];
    document.querySelector('#ms-marginals').innerHTML = groups.map((group) => `
      <article>
        <strong>${group.title}</strong>
        ${Object.entries(group.values).map(([value, probability]) => `
          <div class="trajectory-marginal-row">
            <div><span>${labels[group.variable][value]}</span><b>${percent(probability, 1)}</b></div>
            <span><i style="--trajectory-marginal:${percent(probability, 1)}"></i></span>
            ${group.raw ? `<small>local ${percent(group.raw[value], 1)}</small>` : ''}
          </div>`).join('')}
      </article>`).join('');
  };

  const updateInputLabels = () => {
    document.querySelector('#ms-route-value').textContent = `${inputs.route.value}%`;
    document.querySelector('#ms-fraud-value').textContent = `${inputs.fraud.value}%`;
    document.querySelector('#ms-urgency-value').textContent = `${inputs.urgency.value}%`;
    document.querySelector('#ms-handling-value').textContent = `${inputs.handling.value}%`;
    document.querySelector('#ms-followup-value').textContent = `${inputs.followup.value}%`;
  };

  const render = () => {
    const initial = initialDistributions();
    const policies = activePolicies();
    const inference = enumerateTrajectories(initial, policies);
    const local = greedyTrajectory(initial, policies);
    const map = inference.valid[0];
    const marginals = marginalsFor(inference.valid);

    updateInputLabels();
    document.querySelector('#ms-valid-count').textContent = inference.valid.length;
    document.querySelector('#ms-valid-mass').textContent = inference.z.toFixed(3);
    document.querySelector('#ms-security-marginal').textContent = percent(marginals.route.security, 1);
    document.querySelector('#ms-map-probability').textContent = percent(map.posterior, 1);
    document.querySelector('#ms-local-path').innerHTML = trajectoryMarkup(local);
    document.querySelector('#ms-map-path').innerHTML = trajectoryMarkup(map);

    const localStatus = document.querySelector('#ms-local-status');
    localStatus.textContent = local.violations.length ? `${local.violations.length} conflict${local.violations.length === 1 ? '' : 's'}` : 'Coherent';
    localStatus.classList.toggle('is-coherent', local.violations.length === 0);
    document.querySelector('#ms-local-violations').innerHTML = local.violations.length
      ? local.violations.map((violation) => `<li>${policyLabels[violation]}</li>`).join('')
      : '<li class="no-conflict">The greedy trajectory satisfies every active policy</li>';
    document.querySelector('#ms-map-caption').innerHTML = `Posterior probability <b>${percent(map.posterior, 1)}</b>. Its first decision aggregates the probability of every valid downstream continuation.`;

    renderRankedTrajectories(inference.valid);
    renderMarginals(marginals, initial);
  };

  Object.values(inputs).forEach((input) => input.addEventListener('input', () => {
    document.querySelectorAll('[data-trajectory-preset]').forEach((button) => button.classList.remove('active'));
    render();
  }));

  document.querySelectorAll('[data-trajectory-policy]').forEach((input) => input.addEventListener('change', render));

  document.querySelectorAll('[data-trajectory-preset]').forEach((button) => {
    button.addEventListener('click', () => {
      Object.entries(presets[button.dataset.trajectoryPreset]).forEach(([name, value]) => { inputs[name].value = value; });
      document.querySelectorAll('[data-trajectory-preset]').forEach((candidate) => candidate.classList.toggle('active', candidate === button));
      render();
    });
  });

  render();
}
