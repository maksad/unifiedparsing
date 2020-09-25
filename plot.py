import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

def get_dataframe_from_history_data(path_to_history):
    history_data = torch.load(path_to_history)
    df_train = pd.DataFrame(history_data)
    df_train = df_train.where(pd.notnull(df_train), None)
    return df_train


def get_smoothed_data(data, beta=0.98, strict=False):
    if len(data) <= 1:
        return data

    # get smoothed data with exponentially weighed averages
    data = np.array(data)
    indice = data != None  # noqa: E711
    data_continuous = data[indice]

    smoothed_data = []
    avg_value = 0.
    batch_num = 0
    for i, value in enumerate(data_continuous):
        batch_num += 1
        avg_value = beta * avg_value + (1 - beta) * value
        smoothed_value = avg_value / (1 - beta**batch_num)
        smoothed_data.append(smoothed_value)

    if not strict:
        data[indice] = smoothed_data
    else:
        data = smoothed_data
    return data



def plot_history(path, path_to_save=None):
    df_train = get_dataframe_from_history_data(path)

    column_names = ['object', 'part', 'scene', 'material', 'total']
    plot_columns(
        df_train,
        ['encoder', 'decoder'],
        title='Learning Rates (log scale)',
        prefix='lr',
        fmt='sci',
        path_to_save=f'{path_to_save}_fig_lr' if path_to_save else None,
        figsize=(10, 5),
        shape=(1, 2),
        position=[(0, 0), (0, 1)],
        row_col_span=[(1, 1), (1, 1)],
        smooth_curve=False,
        yscale='log'
    )

    figsize = (20, 8)
    shape = (2, 7)
    position = [(0, 0), (0, 2), (1, 0), (1, 2), (0, 4)]
    row_col_span = [(1, 2), (1, 2), (1, 2), (1, 2), (2, 3)]

    plot_columns(
        df_train,
        column_names,
        df_test=None,
        title='Accuracy',
        prefix='accuracy',
        path_to_save=f'{path_to_save}_fig_acc' if path_to_save else None,
        figsize=figsize,
        shape=shape,
        position=position,
        row_col_span=row_col_span,
    )

    plot_columns(
        df_train,
        column_names,
        df_test=None,
        title='Losses',
        prefix='loss',
        path_to_save=f'{path_to_save}_fig_loss' if path_to_save else None,
        figsize=figsize,
        shape=shape,
        position=position,
        row_col_span=row_col_span
    )


def plot_texture_history(path, path_to_save=None):
    df = get_dataframe_from_history_data(path)
    column_names = ['lr_decoder', 'accuracy', 'loss']
    plot_columns(
        df,
        column_names,
        title="Texture's training parameters",
        fmt='sci',
        path_to_save=f'{path_to_save}_fig_texture' if path_to_save else None
    )


def plot_columns(
    df_train, columns, figsize, shape, position, row_col_span, df_test=None, title=None,
    prefix=None, path_to_save=None, fmt=None, smooth_curve=True, mIoU_column=None, yscale=None
):
    plt.style.use('seaborn')
    fig = plt.figure(figsize=figsize)
    for i, name in enumerate(columns):
        column = prefix + '.' + name if prefix else name
        y_train = df_train[column].to_numpy()
        x_train = np.arange(1, len(y_train) + 1)

        ax = plt.subplot2grid(
            shape, position[i], rowspan=row_col_span[i][0], colspan=row_col_span[i][1], title=name
        )
        if smooth_curve:
            ax.plot(x_train, y_train, alpha=0.25, c='C0')
            label = 'train top-1' if name == 'scene' else f'train {prefix}'
            ax.plot(x_train, get_smoothed_data(y_train, beta=0.99), label=label, c='C0')

            if mIoU_column:
                mIoU_names = [name.split('.')[1] for name in mIoU_column]
                if name in mIoU_names:
                    y_train_mIoU = df_train[f'mIoU.{name}'].to_numpy()
                    x_train_mIoU = np.arange(1, len(y_train_mIoU) + 1)
                    ax.plot(
                        x_train_mIoU, get_smoothed_data(y_train_mIoU, beta=0.99), ls='--',
                        label='train mIoU', c='green'
                    )
        else:
            ax.plot(x_train, y_train, c='C0')

        if fmt == 'sci':
            ax.ticklabel_format(style='sci', scilimits=(-2, 3), axis='y')

        if df_test is not None:
            y_test = df_test[column].to_numpy()
            y_test_smooth = get_smoothed_data(y_test, beta=0.5)
            x_test = df_test.epoch.to_numpy(dtype=int) * df_test.episode.to_numpy(dtype=int)
            if len([i for i in y_test if i]) > 1:
                ax.plot(x_test, y_test, 'r:', lw=1, ms=4, alpha=0.25)
                if smooth_curve:
                    label = 'test top-1' if name == 'scene' else f'test {prefix}'
                    ax.plot(x_test, y_test_smooth, 'r:', marker='o', lw=1, ms=4, label=label)

                    if mIoU_column:
                        mIoU_names = [name.split('.')[1] for name in mIoU_column]
                        if name in mIoU_names:
                            y_test_mIoU = df_test[f'mIoU.{name}'].to_numpy()
                            y_test_mIoU_smooth = get_smoothed_data(y_test_mIoU, beta=0.5)
                            x_test_mIoU = (
                                df_test.epoch.to_numpy(dtype=int) *
                                df_test.episode.to_numpy(dtype=int)
                            )
                            ax.plot(
                                x_test_mIoU, y_test_mIoU_smooth, ls='-.', c='orange', marker='^',
                                lw=1, ms=4, label='test mIoU'
                            )
            ax.legend()

        if yscale:
            ax.set_yscale(yscale)

    if title:
        fig.suptitle(title, y=1.02, fontweight='bold')

    fig.tight_layout()
    if path_to_save:
        svg_path = path_to_save.split('/')
        svg_file_name = f'svg_{svg_path[-1]}'
        svg_path = '/'.join(svg_path[:-1])

        plt.savefig(f'{path_to_save}.png', bbox_inches='tight')
        plt.savefig(f'{svg_path}/{svg_file_name}.svg', bbox_inches='tight')
    else:
        plt.show()
    plt.close()
