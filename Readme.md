<!-- Reademe template from https://github.com/othneildrew/Best-README-Template/blob/master/BLANK_README.md -->


<!--
*** Thanks for checking out this README Template. If you have a suggestion that would
*** make this better, please fork the repo and create a pull request or simply open
*** an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** StatResp, EMS_DSS, twitter_handle, geoffrey.a.pettet@vanderbilt.edu
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!--
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->


<!-- PROJECT LOGO -->
<br />
<p align="center">
<!--  <a href="https://github.com/StatResp/EMS_DSS">
    <img src="" alt="Logo">
  </a> -->

  <h3 align="center">Emergency Response Decision Support System and Simulation Framework</h3>

 <!-- <p align="center">
    YOUR_SHORT_DESCRIPTION
    <br />
    <a href="https://github.com/StatResp/EMS_DSS"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/StatResp/EMS_DSS">View Demo</a>
    ·
    <a href="https://github.com/StatResp/EMS_DSS/issues">Report Bug</a>
    ·
    <a href="https://github.com/StatResp/EMS_DSS/issues">Request Feature</a>
  </p> -->
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

This project implements an EMS responder allocation decision support system, as well as a discrete event 
simulator to evaluate the system.   

***Disclaimer - This is a work in progress (the project and the readme)***


### Built With

* [Python3](https://python.org)
* [Vis Framework (Pending)](https://link)

## Getting Started

1. Install required python packages - found in requirements.txt

2. Clone the repo
```sh
git clone https://github.com/StatResp/EMS_DSS.git
```

3. Update the configuration files to match your environment structure to point the code to the correct data 
directories. Config files include: 
 * scenarios/gridworld_example/definition/grid_world_consts.py

<!--
2. Clone the repo
```sh
git clone https://github.com/StatResp/EMS_DSS.git
```
2. Install NPM packages
```sh
npm install
```
-->


## Project Structure

The framework consists of several different components, including decision makers (high and low level), an 
ems simulator, and several experimental scenarios. 

Currently, only the ems simulator and ems environmental dynamics are implemented. To run an example simulation
with toy data, run scenarios/gridworld_example/testing.py. This example takes place in a gridworld with 5 depots
and 3 responders. There are a few manual incidents added to the event queue. It simply sends the nearest available
responder to an incident when the incident occurs. 



<!-- _For more examples, please refer to the [Documentation](https://example.com)_ -->

<!--

## Roadmap

See the [open issues](https://github.com/StatResp/EMS_DSS/issues) for a list of proposed features (and known issues).



## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## License

Distributed under the MIT License. See `LICENSE` for more information.

-->

<!-- CONTACT -->
## Contact

Your Name - [@](https://twitter.com/) - geoffrey.a.pettet@vanderbilt.edu

Project Link: [https://github.com/StatResp/EMS_DSS](https://github.com/StatResp/EMS_DSS)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* []()
* []()
* []()





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/StatResp/repo.svg?style=flat-square
[contributors-url]: https://github.com/StatResp/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/StatResp/repo.svg?style=flat-square
[forks-url]: https://github.com/StatResp/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/StatResp/repo.svg?style=flat-square
[stars-url]: https://github.com/StatResp/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/StatResp/repo.svg?style=flat-square
[issues-url]: https://github.com/StatResp/repo/issues
[license-shield]: https://img.shields.io/github/license/StatResp/repo.svg?style=flat-square
[license-url]: https://github.com/StatResp/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/StatResp
[product-screenshot]: images/screenshot.png