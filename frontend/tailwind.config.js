/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          dark: '#0f172a',    // Slate 900
          light: '#f8fafc',   // Slate 50
          accent: '#10b981',  // Emerald 500
          warning: '#f59e0b', // Amber 500
          danger: '#ef4444'   // Red 500
        }
      }
    },
  },
  plugins: [],
}
