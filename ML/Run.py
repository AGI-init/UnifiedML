# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from minihydra import get_args, instantiate, interpolate, Args  # minihydra conveniently and cleanly manages sys args

from Utils import MT, init, adaptive_shaping, MP, save, load


@get_args(source='Hyperparams/args.yaml')  # Hyper-param arg files located in ./Hyperparams
def main(args):
    if args.multi_task:
        return MT.launch(args.multi_task)  # Handover to multi-task launcher

    # Set random seeds, device
    init(args)

    # Train, test environments
    env = instantiate(args.environment) if args.train_steps else None
    generalize = instantiate(args.environment, train=False, seed=args.seed + 1234)

    # Update args
    interpolate(args, _obs_spec_=Args(generalize.obs_spec), _action_spec_=Args(generalize.action_spec))

    # Experience replay
    replay = instantiate(args.replay) if args.train_steps else args.replay

    # Agent
    agent = load(args.load_path, args.device, args.agent) if args.load \
        else instantiate(args.agent, **adaptive_shaping(obs_spec=args.agent.obs_spec,
                                                        action_spec=args.agent.action_spec)).to(args.device)

    # Synchronize multi-task models (if exist)
    agent = MT.unify_agent_models(agent, args.agent, args.device, args.load and args.load_path)

    # Logger / Vlogger
    logger = instantiate(args.logger, witness=agent)
    vlogger = instantiate(args.vlogger) if args.log_media else None

    train_steps, replay.epoch = args.train_steps + agent.step, agent.epoch

    # Start
    converged = training = args.train_steps == 0
    while True:
        # Evaluate
        if converged or (args.evaluate_per_steps and agent.step % args.evaluate_per_steps == 0):

            for _ in range(args.generate or args.evaluate_episodes):
                if not agent.step or agent.step > logger.step:
                    exp, log, vlog = generalize.rollout(agent.eval(),  # agent.eval() just sets agent.training to False
                                                        vlog=args.log_media)

                    logger.eval().log(log, step=agent.step)

            logger.eval().dump(exp if converged else None)  # Dump logs. At convergence, also dump eval experiences

            if args.log_media:
                vlogger.dump(vlog, f'{agent.step}')

        if args.plot_per_steps and (agent.step + 1) % args.plot_per_steps == 0 and not args.generate or converged:
            instantiate(args.plotting)  # TODO show=converged and args.show -> plot in web browser or window-pane

        if converged:
            break

        # Rollout
        experiences, log, _ = env.rollout(agent.train(), steps=1)  # agent.train() just sets agent.training to True
        replay.add(experiences)

        if env.episode_done and args.log_per_episodes and (agent.episode - 2 * args.offline) % args.log_per_episodes \
                == 0 or args.log_per_steps and agent.step > 1 and (agent.step - args.offline) % args.log_per_steps == 0:
            logger.mode('Train' if training else 'Seed').log(log, dump=True)

        converged = agent.step >= train_steps
        training = training or agent.step > args.seed_steps and len(replay) > replay.partitions - 1 or replay.offline

        # Train agent
        if training and (args.learn_per_steps and agent.step % args.learn_per_steps == 0 or converged):

            for _ in range(args.learn_steps_after if converged else args.learn_steps):
                log = Args(time=None, step=None, frame=None, episode=None, epoch=None)
                agent.learn(replay, log)  # Learn
                logger.train().re_witness(log, agent, replay)
                if (args.log_per_episodes or args.log_per_steps) and args.agent.log:
                    logger.log(log)

                if args.mixed_precision:
                    MP.update()  # For training speedup via automatic mixed precision

        if training and args.save_per_steps and agent.step % args.save_per_steps == 0 or (converged and args.save):
            save(args.save_path, agent, args.agent, 'birthday', 'frame', 'step', 'episode', 'epoch')

        if training and args.load_per_steps and agent.step % args.load_per_steps == 0:
            agent = load(args.load_path, args.device, args.agent, ['birthday', 'frame', 'step', 'episode', 'epoch'])

    return agent, replay


if __name__ == '__main__':
    main()
