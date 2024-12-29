---
title: "Decentralised Compute"
date: 2023-06-07T00:13:25+10:00
draft: true
---

Chainlink Functions is a service that allows your smart contracts to access a trust-minimized compute infrastructure. Your smart contract sends your code to a Decentralized Oracle Network (DON), and each oracle within the DON executes the same code in a serverless environment. The DON then aggregates all the independent runs and returns the final result to your smart contract. This code can range from simple computations to fetching data from API providers​1​.

Chainlink Functions is a self-service solution, meaning that you are responsible for independently reviewing any JavaScript code that you write and submit in a request. This includes API dependencies that you send to be executed by Chainlink Functions. It's worth noting that community-created JavaScript code examples might not be audited, so you must review this code independently before using it. The service is offered "as is" and "as available" without any conditions or warranties. Chainlink Labs, the Chainlink Foundation, and Chainlink node operators are not responsible for unintended outputs generated by Functions due to errors in the submitted JavaScript code or issues with API dependencies​1​.

Chainlink Functions supports a wide variety of use cases. You can use it to:

Connect to any public data.
Transform public data before consumption.
Connect to password-protected data sources, from IoT devices to enterprise resource planning systems.
Connect to an external decentralized database to facilitate off-chain processes for a dApp or build a low-cost governance voting system.
Build complex hybrid smart contracts by connecting to your Web2 applications.
Fetch data from almost any Web2 system such as AWS S3, Firebase, or Google Cloud Storage​1​.
As of the time of writing, Chainlink Functions is available on testnet only as a limited BETA preview. While in this phase, developers must follow best practices and avoid using the BETA for any production application or securing any value. Chainlink Functions is likely to evolve and improve, and breaking changes may occur while the service is in BETA​1​.

https://docs.chain.link/chainlink-functions