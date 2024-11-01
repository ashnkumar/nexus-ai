![High Level Overview](https://i.imgur.com/vPkMjFZ.png)
**TLDR** Nexus AI is a system that combines privacy-preserving decentralized machine learning with autonomous AI agents using NEAR smart contracts and Calimero's applications. It allows crypto traders to collaboratively train machine learning models on their private trading data in a secure way (data never leaves their system) and provide the output of that model to an AI agent for it to reason and take trading actions.

#### Problem(s) We're Solving
Many users see the value in combining proprietary data to build user-owned AI. In crypto trading, for example, expert traders could merge strategies to boost the predictive accuracy of ML models. But sharing data risks privacy and exposes competitive advantages.

Limited access to advanced AI models also makes it harder for solo traders, who can't match larger institutions equipped with cutting-edge tools like LLMs and AI-driven agents. 

We need a way to enable people to leverage their own individual data and work together to augment their use of AI, while maintaining their privacy and security.

#### Overview of Nexus
Nexus facilitates **privacy-preserving** collaborative AI training among a community of users (in our case, crypto traders), leveraging NEAR and Calimero's capabilities. 

It combines 3 key technologies:
1. **Web3**: Smart contracts for capital pooling, distributed training coordination, and automated profit distribution
2. **Federated Learning**: A way to train models with end users' data in a way where the data never has to leave the users' systems
3. **AI Agents**: Leveraging LLMs that can reason and using the model trained in step 2, the AI agent can take actions autonomously on behalf of the group of users; in our case, making crypto trades. This way we're combining the quantitative prediction power of ML models with the reasoning of AI agents

#### How Nexus Works

![Technical Overview](https://i.imgur.com/L43G6PD.png)

**Local model training with privacy preservation:** Each trader trains a local LSTM machine learning model on their own historical trading data. Only the encrypted model updates are shared with the network, not the raw data, ensuring privacy.

**Secure aggregation of model updates for a Global Model:** Encrypted model training weights updates are securely stored using Calimero's protocol and storage, and then combined into a trained global model. In our case, we're using LSTM neural networks to predict token price movement. This process ensures that no individual trader's data is exposed during aggregation - all data stays on the users' systems.

**Contribution analysis and profit distribution:** We calculate each trader's contribution to the global model's accuracy. NEAR's smart contracts are used to automate profit distribution, rewarding traders proportionally based on their contributions.

**AI Agent for enhanced trading:** An AI agent powered by an LLM uses the global model's predictions and real-time market data to make informed trading decisions. We used GPT-4 for simplicity but the goal is to use open-source LLMs like Llama as soon as possible. It can then take actions (make trades) through the function calling capabilities of the LLM. The agent executes trades autonomously, aiming to outperform individual strategies. 

**Fair and transparent profit sharing:** Profits generated by the AI agent are distributed to traders using NEAR's blockchain through a smart contract, ensuring transparency and trust. Traders can withdraw their earnings at any time.

#### Why NEAR and Calimero?

Our project leverages the strengths of both NEAR and Calimero to provide a secure, scalable, and user-friendly platform.

**Calimero's privacy-preserving framework:** Calimero Network enables secure, privacy-focused applications. It allows us to implement federated learning where data remains on the host machine and encrypted model updates are shared on Calimero storage and aggregated offline without exposing raw data, preserving each trader's privacy.

**NEAR's scalable and user-friendly Blockchain:**
NEAR offers a sharded, proof-of-stake blockchain with low fees and high throughput. We created NEAR smart contracts to facilitate the group's capital pooling and automated profit distribution.

#### What's Next

Our current focus is on crypto traders, but the framework we've developed has broader applications. It can apply to _ANY_ community of people that want to combine their data to leverage the full power of AI.

**Expansion to other domains:** The platform can be extended to any collaborative AI training where data privacy is crucial, such as healthcare, finance, or supply chain optimization. Organizations can use this framework to build powerful AI models without compromising sensitive data.

**Enhancing the AI agent:** Incorporate more advanced AI agents with improved reasoning and the ability to work with other AI agents to make decisions on actions to take. This also means expanding the universe of actions it can take.

**Community and ecosystem growth:** Open-source the project to encourage contributions from the developer community. Foster partnerships with other projects in the NEAR ecosystem to expand functionality and impact.

#### Hackathon Criteria

| Criteria | Explanation |
|----------|-------------|
| **Technological Implementation** | Our project combines multiple features of NEAR, Calimero, and extensive Python ML development. We implement federated learning, secure data sharing, and decentralized profit distribution |
| **Design** | We worked hard to design an experience that's seamless for users. Our goal is to abstract away advanced concepts like ML training and AI agents so more people can join to collaborate to create user-owned AI |
| **Potential Impact** | The project is open-source and has the potential to significantly impact the NEAR ecosystem by introducing a novel use case that combines AI, blockchain, and privacy. It can attract new users and developers interested in privacy-preserving AI solutions |
| **Quality of the Idea** | We loved combining federated learning, AI agents, and blockchain to address a real-world problem in trading and AI model training |
| **Project Sustainability** | We have quite a bit to go in the future mapped out. The project is completely open-sourced and we can quickly allow crypto traders and other groups to set up their own version of this group-based AI training using the blockchain for collaboration. They can use their own LLM models (or API keys) so it reduces the cost for us to use generative LLMs |
