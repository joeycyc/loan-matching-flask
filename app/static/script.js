"use-strict";

// This file is not used

console.log("JavaScript Loaded!");

const resultProject = document.getElementById("result-display-project");
const resultLoan = document.getElementById("result-display-loan");
const resultDownload = document.getElementById("result-display-download");

resultProject.addEventListener("DOMSubtreeModified", function () {
  console.log("haha");
  resultDownload.innerHTML = "";
  // while (resultDownload.firstChild) {
  //   resultDownload.removeChild(resultDownload.lastChild);
  // }
});

// // Options for the observer
// const config = { attributes: true, childList: true, subtree: true };

// // Callback function to execute when mutations are observed
// const callbackProject = (mutationList, observer) => {
//   for (const mutation of mutationList) {
//     if (mutation.type === "childList") {
//       console.log("inside");
//       resultLoan.innerHTML = "";
//       resultDownload.innerHTML = "";
//     }
//   }
// };

// // Create an observer instance linked to the callback function
// const observerProject = new MutationObserver(callbackProject);

// // Start observing the target node for configured mutations
// observerProject.observe(resultProject, config);

// Stop observing
// observerProject.disconnect();
