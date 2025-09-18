---
---

// Dark mode functionality
(function() {
  'use strict';

  // Constants
  const STORAGE_KEY = 'ai-blog-theme';
  const THEME_LIGHT = 'light';
  const THEME_DARK = 'dark';

  // DOM elements
  let toggleButton;
  let toggleIcon;
  let toggleText;

  // Initialize dark mode
  function initDarkMode() {
    // Create toggle button
    createToggleButton();

    // Set initial theme
    const savedTheme = localStorage.getItem(STORAGE_KEY);
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const initialTheme = savedTheme || (prefersDark ? THEME_DARK : THEME_LIGHT);

    setTheme(initialTheme);

    // Listen for system theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
      if (!localStorage.getItem(STORAGE_KEY)) {
        setTheme(e.matches ? THEME_DARK : THEME_LIGHT);
      }
    });
  }

  // Create toggle button
  function createToggleButton() {
    toggleButton = document.createElement('button');
    toggleButton.className = 'dark-mode-toggle';
    toggleButton.setAttribute('aria-label', 'Toggle dark mode');
    toggleButton.setAttribute('title', 'Toggle dark mode');

    toggleIcon = document.createElement('span');
    toggleIcon.className = 'toggle-icon';
    toggleIcon.setAttribute('aria-hidden', 'true');

    toggleText = document.createElement('span');
    toggleText.className = 'toggle-text';

    toggleButton.appendChild(toggleIcon);
    toggleButton.appendChild(toggleText);

    // Add click event
    toggleButton.addEventListener('click', toggleTheme);

    // Add to DOM
    document.body.appendChild(toggleButton);
  }

  // Set theme
  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);

    // Update button
    if (theme === THEME_DARK) {
      toggleIcon.textContent = '☀️';
      toggleText.textContent = 'Light';
    } else {
      toggleIcon.textContent = '🌙';
      toggleText.textContent = 'Dark';
    }

    // Save to localStorage
    localStorage.setItem(STORAGE_KEY, theme);

    // Update meta theme-color for mobile browsers
    updateMetaThemeColor(theme);

    // Dispatch custom event
    document.dispatchEvent(new CustomEvent('themechange', {
      detail: { theme: theme }
    }));
  }

  // Toggle theme
  function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === THEME_DARK ? THEME_LIGHT : THEME_DARK;

    // Add animation class
    toggleButton.style.transform = 'scale(0.95)';
    setTimeout(() => {
      toggleButton.style.transform = '';
    }, 150);

    setTheme(newTheme);
  }

  // Update meta theme-color for mobile browsers
  function updateMetaThemeColor(theme) {
    let themeColorMeta = document.querySelector('meta[name="theme-color"]');

    if (!themeColorMeta) {
      themeColorMeta = document.createElement('meta');
      themeColorMeta.name = 'theme-color';
      document.head.appendChild(themeColorMeta);
    }

    const themeColors = {
      [THEME_LIGHT]: '#ffffff',
      [THEME_DARK]: '#0d1117'
    };

    themeColorMeta.content = themeColors[theme];
  }

  // Add keyboard shortcut (Ctrl/Cmd + Shift + D)
  function addKeyboardShortcut() {
    document.addEventListener('keydown', function(e) {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'D') {
        e.preventDefault();
        toggleTheme();
      }
    });
  }

  // Animate elements when theme changes
  function addThemeChangeAnimation() {
    document.addEventListener('themechange', function() {
      // Add subtle animation to all elements with transitions
      const elements = document.querySelectorAll('*');
      elements.forEach(el => {
        if (getComputedStyle(el).transition) {
          el.style.transition = 'all 0.3s ease';
        }
      });
    });
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
      initDarkMode();
      addKeyboardShortcut();
      addThemeChangeAnimation();
    });
  } else {
    initDarkMode();
    addKeyboardShortcut();
    addThemeChangeAnimation();
  }

  // Expose functions globally for debugging
  window.darkMode = {
    setTheme: setTheme,
    toggleTheme: toggleTheme,
    getCurrentTheme: function() {
      return document.documentElement.getAttribute('data-theme');
    }
  };

})();