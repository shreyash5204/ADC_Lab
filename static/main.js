const themeToggle = document.getElementById('theme-toggle');
const body = document.body;
const nav = document.querySelector("nav");

// const checkbox = document.getElementById("checkbox");

// checkbox.addEventListener("change", () => {
//   body.classList.toggle('dark-mode');
//   nav.classList.toggle('dark-mode');
//   document.body.classList.toggle("dark")
// });

var isDark = false;

const themeButton = document.querySelector('.light-mode-button');

themeButton.addEventListener('click', ()=>{
  isDark = !isDark;
  if(isDark)
  themeButton.innerHTML = '<img src="../static/images/AM_MAIN/icons8-sun-24.png" alt="button"/>';
  else
  themeButton.innerHTML = '<img src="../static/images/AM_MAIN/icons8-moon-48 (1).png" alt="button"/>'

  body.classList.toggle('dark-mode');
  nav.classList.toggle('dark-mode');
  document.body.classList.toggle("dark")
});
