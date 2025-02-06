const primaryNav = document.querySelector('#primary-navigation');
const navToggle = document.querySelector('#mobile-nav-toggle');
const lowerNavHr = document.querySelector('#lower-nav-hr');
const spaceForHeader = document.querySelector('#space-for-header');
const logoAndButton = document.querySelector('#logo-and-button');

navToggle.addEventListener('click', () => {
    if (primaryNav.getAttribute('data-visible') === "false") {
        primaryNav.setAttribute('data-visible', 'true');
        navToggle.setAttribute('aria-expanded', 'true');
    } else {
        primaryNav.setAttribute('data-visible', 'false');
        navToggle.setAttribute('aria-expanded', 'false')
    }
    console.log(primaryNav.getAttribute('data-visible'));
});

const header = document.querySelector("#header");
const headerLogo = document.querySelector("#nav-logo");

window.addEventListener('scroll', () => {
    console.log(window.scrollY);
    if (window.scrollY > 200) {
        header.classList.add('header-fixed-scroll');
        headerLogo.classList.add('nav-logo-scroll');
    } else {
        header.classList.remove('header-fixed-scroll');
        headerLogo.classList.remove('nav-logo-scroll');
    }
});