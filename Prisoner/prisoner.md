# Multi-Round *Prisoners' Dilemma*

*Refer to CS50-AI  and build an "AI prisoner" with machine learning*

- **Basic Rules for every round(every game contains many rounds)**

  > if both prisoners choose to confess, both prisoners will be sentenced to 8 years in prison.
  >

  > if both prisoners choose to remain silent, both prisoners will be sentenced to 1 year in prison.
  >

  > if one prisoner chooses to confess and the other chooses to remain silent, the one who confesses will be set free and the other will be sentenced to 10 years in prison.
  >

  > 0 means remain silent
  >

  > 1 means confess
  >
- **Some special players**

  1. choose to be silent until "betrayed" 5 times
  2. choose to simulate(copy) the other player
- **Result**

  - After 10000 training games, AI can fully grasp the rules of the game and its opponents, and choose the best strategy to minimize its own sentence.
  - ```
    (Play against P1)
    Game  10000
    0 -10
    0 -10
    0 -10
    0 -10
    -1 -1
    -1 -1
    -1 -1
    -1 -1
    -1 -1
    0 -10
    -5 -55
    ```
  - ```
    (Play against P2)
    Game  10000
    -1 -1
    -1 -1
    -1 -1
    -1 -1
    -1 -1
    -1 -1
    -1 -1
    -1 -1
    -1 -1
    0 -10
    -9 -19
    ```
  - ```
    (Play against itself)
    AI training round:  99997
    -24 -24
    AI training round:  99998
    -24 -24
    AI training round:  99999
    -24 -24
    AI training round:  100000
    -1 -1
    -8 -8
    -8 -8
    -1 -1
    -1 -1
    -1 -1
    -1 -1
    -1 -1
    -1 -1
    -1 -1
    -24 -24
    AI training finished
    ```
- **Some thoughts**

  - By a plenty of experiences, AI can figure out a formulated strategy and thus minimize its own cost.  ~~The same in life.~~
  - However, for two much intelligent players, they actually can't get the optimal  result because they always try to get the best for themself and  even cunningly choose to betray. Maybe that's another special form of *Nash Equilibrium*. Above all, how essential the **cooperations and trust** in life!
