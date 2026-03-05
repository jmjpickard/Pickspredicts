/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,js,ts}"],
  theme: {
    extend: {
      colors: {
        gold: "#c8a951",
        racing: {
          green: "#1a472a",
          cream: "#faf8f0",
        },
      },
    },
  },
  plugins: [],
};
