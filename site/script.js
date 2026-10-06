const navToggle = document.querySelector('.nav-toggle');
const primaryNav = document.querySelector('.primary-nav');

navToggle?.addEventListener('click', () => {
  const isOpen = navToggle.getAttribute('aria-expanded') === 'true';
  navToggle.setAttribute('aria-expanded', String(!isOpen));
  primaryNav?.classList.toggle('open', !isOpen);
});

primaryNav?.querySelectorAll('a').forEach((link) => {
  link.addEventListener('click', () => {
    navToggle?.setAttribute('aria-expanded', 'false');
    primaryNav.classList.remove('open');
  });
});

const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        revealObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.12 }
);

document.querySelectorAll('.reveal').forEach((element) => revealObserver.observe(element));

const filterButtons = document.querySelectorAll('.filter-button');
const projectCards = document.querySelectorAll('.project-card');

filterButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const filter = button.dataset.filter;
    filterButtons.forEach((candidate) => candidate.classList.toggle('active', candidate === button));

    projectCards.forEach((card) => {
      const categories = (card.dataset.category || '').split(' ');
      card.classList.toggle('hidden', filter !== 'all' && !categories.includes(filter));
    });
  });
});

const copyButton = document.querySelector('#copy-code');

copyButton?.addEventListener('click', async () => {
  const target = document.querySelector(`#${copyButton.dataset.copyTarget}`);
  if (!target) return;

  try {
    await navigator.clipboard.writeText(target.innerText);
    copyButton.textContent = 'Copied';
    window.setTimeout(() => {
      copyButton.textContent = 'Copy';
    }, 1600);
  } catch {
    copyButton.textContent = 'Select code';
  }
});

const flowDemo = document.querySelector('#decisionflow-demo');
const flowSteps = flowDemo?.querySelectorAll('[data-flow-step]') || [];
const flowStages = flowDemo?.querySelectorAll('[data-flow-stage]') || [];
const flowLive = flowDemo?.querySelector('.flow-live');
const flowStepper = flowDemo?.querySelector('.flow-stepper');
const flowMessages = [
  'Connecting decision models',
  'Reading the declarative business workflow',
  'Compiling the decision structure',
  'Running structured inference',
];

let activeFlowStep = 0;

const renderFlowStep = (step) => {
  if (flowDemo) flowDemo.dataset.activeStep = String(step);
  flowStepper?.style.setProperty('--step-translate', `calc(${step * 100}% + ${step * 7}px)`);
  flowSteps.forEach((item) => {
    item.classList.toggle('is-active', Number(item.dataset.flowStep) === step);
  });
  flowStages.forEach((item) => {
    item.classList.toggle('is-active', Number(item.dataset.flowStage) === step);
  });
  if (flowLive) flowLive.textContent = flowMessages[step];
};

renderFlowStep(activeFlowStep);

if (flowDemo && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  window.setInterval(() => {
    activeFlowStep = (activeFlowStep + 1) % flowMessages.length;
    renderFlowStep(activeFlowStep);
  }, 3000);
}
