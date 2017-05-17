import logging
import os
import pyvsem.neuralnets.prepare as prepare

import matplotlib
matplotlib.use('Agg')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s: %(message)s')
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)


def main(net_number, colors=1, width=200, data_set=None, aug=False, task='multi_label', batch_size=128):
    image_shape = (colors, width, width)
    image_size = (width, width)

    net = prepare.get_net_module(net_number)
    data_dir = prepare.get_data_dir(data_set)
    data_provider = prepare.get_data_provider(data_dir, data_set, task)

    current_run = 'net{}_{}_{}x{}x{}'.format(net_number, task, colors, width, width)
    if aug:
        current_run += '_aug'
    else:
        current_run += '_noaug'

    current_run += "_{}".format(batch_size)

    log.info("Current run: '{}'".format(current_run))

    x_train, x_test, y_train, y_test = prepare.get_data(data_dir, task, data_provider)

    train_iterator = prepare.get_train_iterator(image_size, image_shape, batch_size, aug)
    test_iterator = prepare.get_test_iterator(image_size, image_shape, batch_size)

    on_epoch_finished = prepare.get_on_epoch_finished(data_dir, current_run)

    nnet = net.create_net(
        input_shape=image_shape,
        y=data_provider.get_all_tags(),
        train_iterator=train_iterator,
        test_iterator=test_iterator,
        max_epochs=100,
        on_epoch_finished=on_epoch_finished
    )

    print(x_train, y_train)

    nnet.fit(x_train, y_train)

    log.info("Net '{}' successfully fit".format(current_run))

    # save_pickle(nnet, net_pickle)
    nnet.save_params_to(os.path.join(data_dir, "{}_params.pickle".format(current_run)))

    log.info("Net '{}' stored as parameters".format(current_run))


if __name__ == '__main__':
    parser = prepare.get_args_parser()

    args = parser.parse_args()

    log.info("Running with the following arguments: {}".format(args))

    main(net_number=args.net,
         colors=args.colors,
         width=args.width,
         data_set=args.data_set,
         aug=args.aug,
         task=args.task,
         batch_size=args.batch_size)
