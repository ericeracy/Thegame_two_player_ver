# Thegame_two_player_ver
This repository is part of the master project *Multi-agent cooperation and information exchange in partially observable games* (MACIE). We conduct research for agent cooperation and communication using two-player Atari Pong (code in another repo) and NSV Thegame. Especially for Thegame, we explore how fuzzy signals works with the help of three game modes: full information, no communication, and fuzzy signal exchange.

## What inside
This code repo consists of the emulator of the board game [Thegame Quick & Easy](https://nsv-games.com/the-game-quick-easy/) (referred as *Thegame* afterwards) proposed by NSV, the DQN models for two-player Thegame within three modes of agents's communication. 

## How to use
Both `run.sh` and `run_and_test.sh` can be used for runnning and will generate a report file in the `data` (autocreated if not exist) directory. The only different is `run_and_test.sh` includes the test procedure and will give the report regarding how models work on testing.

## Mode description
Thegame emulator agents have three communication modes: no communication, absolute number, fuzzy number, which are introduced in the master project MACIE: 
**No-communication (no-commu):** agents are only aware of their hand cards and the top-most stack cards. They have no idea what cards the partner has. 
**Absolute signal:** agents know the full information of the partner's current hand cards
**Fuzzy quantifiers:** one agent tells the relative size of the hand cards to the partner.

This idea comes from the categories of quantifiers: absolute quantifiers (the exact number: 1, 2, 3, ...) and fuzzy quantifiers (e.g., many, a lot) that are commonly used in our diary life. 
