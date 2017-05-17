import logging
import os
import numpy as np
import sklearn

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import pyvsem.neuralnets.prepare as prepare

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s: %(message)s')
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)


def main(net_number, colors, width, data_set, aug, task, batch_size, pretrain_set, pretrain_run, save=None):
    image_shape = (colors, width, width)
    image_size = (width, width)

    net = prepare.get_net_module(net_number)
    data_dir = prepare.get_data_dir(data_set)
    data_provider = prepare.get_data_provider(data_dir, data_set, task)

    current_run = 'pretrained_net{}_{}_{}x{}x{}'.format(net_number, task, colors, width, width)
    if aug:
        current_run += '_aug'
    else:
        current_run += '_noaug'
    if save is not None:
        current_run += '_n{}'.format(save)
    current_run += "_{}".format(batch_size)

    log.info("Current run: '{}'".format(current_run))

    x_train, x_test, y_train, y_test = prepare.get_data(data_dir, task, data_provider)

    if save is not None:
        tarname = "train-test-split-nr{}.tar.bz2".format(save)
        tarname = os.path.join(data_dir, tarname)
        log.info("Compressing images from train/test to {}".format(tarname))
        import tarfile
        with tarfile.open(tarname, 'w:bz2') as tar:
            for train_img in x_train:
                tar.add(train_img, arcname=os.path.join('train', os.path.basename(train_img)))
            for test_img in x_test:
                tar.add(test_img, arcname=os.path.join('test', os.path.basename(test_img)))

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

    log.info("Net '{}' successfully initialized".format(current_run))

    pretrain_params = os.path.join(data_dir, os.pardir, pretrain_set, "{}_params.pickle".format(pretrain_run))
    log.info("Loading parameters for net from: '{}'".format(pretrain_params))
    nnet.load_params_from(pretrain_params)

    nnet.fit(x_train, y_train)

    log.info("Net '{}' successfully fit".format(current_run))

    # save_pickle(nnet, net_pickle)
    nnet.save_params_to(os.path.join(data_dir, "{}_params.pickle".format(current_run)))

    log.info("Net '{}' stored as parameters".format(current_run))

    preds = nnet.predict(x_test)

    _, _, f1, sup = sklearn.metrics.precision_recall_fscore_support(y_true=y_test, y_pred=preds)
    with open(os.path.join(data_dir, "{}_report.txt".format(current_run)), 'w') as out:
        out.write(
            sklearn.metrics.classification_report(
                y_true=data_provider.inverse_transform(y_test),
                y_pred=data_provider.inverse_transform(preds),
                digits=3)
        )

    _, _, fscore, support = sklearn.metrics.precision_recall_fscore_support(
        y_true=[data_provider.inverse_transform(x) for x in y_test],
        y_pred=[data_provider.inverse_transform(x) for x in preds],
        labels=data_provider.all_tags,
        average=None
    )

    # plot f1 / support
    width = 0.8
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title('F1 score vs. class (Run #{})'.format(save))

    ax.set_xlabel('classes (count of instances)')
    ax.set_ylabel('F1 score')

    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_ylim(0.0, 1.0)

    ax.set_xticks(range(len(fscore)))
    ax.set_xticklabels(
        ["{} ({})".format(data_provider.all_tags[i], sup) for i, sup in sorted(enumerate(support), key=lambda x: x[1])],
        rotation='vertical')
    #ax.set_xlim(width, len(s) + width)

    bar = plt.bar(
        range(len(fscore)),
        [f for f, _ in sorted(zip(fscore, support), key=lambda x: x[1])],
        width=width)

    plt.savefig(os.path.join(data_dir, '{}_f1-p-class.pdf'.format(current_run)),
        bbox_inches='tight',
        transparent=True)



if __name__ == '__main__':
    parser = prepare.get_args_parser()
    parser.add_argument(
        "--pretrain-run",
        help="name of the run from which to extract pretrained parameters",
        type=str,
        required=True
    )
    parser.add_argument(
        "--pretrain-set",
        help="name of the dataset which was used to pretrain",
        type=str,
        required=True
    )

    parser.add_argument(
        "--save",
        help="Whether to store the images from the train/test set",
        type=str
    )

    args = parser.parse_args()

    log.info("Running with the following arguments: {}".format(args))

    main(net_number=args.net,
         colors=args.colors,
         width=args.width,
         data_set=args.data_set,
         aug=args.aug,
         task=args.task,
         batch_size=args.batch_size,
         pretrain_set=args.pretrain_set,
         pretrain_run=args.pretrain_run,
         save=args.save
    )
