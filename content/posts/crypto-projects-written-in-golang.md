---
title: "Crypto Projects Written in Golang"
date: 2022-11-07T22:18:07+11:00
draft: false
tags:
- Blockchain-Development
- Go
---
I first started developing in Golang back in 2012 [upon version 1 release](https://go.dev/blog/go1) & since its initial release, Golang has become quite popular with several major crypto & blockchain projects. This is a quick overview of the projects which have sparked my interest firstly starting with a quote providing some insight into why Golang has had success in the blockchain space.

>Golang is a systems programming language for building networked, distributed services. The main selling point is strong [CSP](https://en.wikipedia.org/wiki/Communicating_sequential_processes) style concurrency support.<br>
>Blockchain apps are networked, distributed services. They benefit greatly from good concurrency support, as getting this wrong can lead to loss of money.
> — <cite>Vladislav Zorov[^1]</cite>

[^1]: Why is Golang considered the best programming language for building a Blockchain App? [Quora Answer from Zorov](https://www.quora.com/Why-is-Golang-considered-the-best-programming-language-for-building-a-Blockchain-App/answer/Vladislav-Zorov) 

### GETH - GO Ethereum 
[GETH](https://github.com/ethereum/go-ethereum) 

### Polygon
[Polygon](https://github.com/maticnetwork/bor) is an Ethereum layer 2

### Cosmos
[Cosmos](https://github.com/cosmos/cosmos-sdk)

### Thorchain
[Thorchain](https://github.com/thorchain/thornode)

Thorchain is a decentralized liquidity protocol built to enable cross-chain token swaps without requiring a trusted intermediary or wrapping tokens. Launched in 2018, Thorchain is built on Tendermint and Cosmos-SDK and allows users to swap tokens between different blockchains in a decentralized, non-custodial manner.   

### Rocket Pool 
[Rocket Pool](https://github.com/rocket-pool/rocketpool) is a decentralised staking pool for Ethereum 2.0

### Chainlink 
[Chainlink](https://github.com/smartcontractkit/chainlink) Chainlink is a decentralized oracle network that enables smart contracts on several blockchains to securely connect and interact with real-world data & external APIs.

Additionally, Chainlink provides a [CCIP (Cross-Chain Interoperability Protocol)](https://chain.link/cross-chain) which provides [cross-chain](https://www.youtube.com/watch?v=6DgnHKTI-EU) communication between public & private blockchains. For example Etherumn & [Swift interoperability](https://www.chainlinkecosystem.com/ecosystem/swift). Which could additionally provide [interoperability into CBDC's](https://www.businesswire.com/news/home/20221005005149/en/). CCIP technology will be able to connect traditional capital markets with trillions of dollars of value into decentralized finance ([DeFi](https://en.wikipedia.org/wiki/Decentralized_finance)). [CCIP is in the final stages of security review](https://chain-reaction.simplecast.com/episodes/chainlink-co-founder-talks-cryptographic-guarantees-in-web3-w-sergey-nazarov-PUnPyiNu?t=23m0s).
