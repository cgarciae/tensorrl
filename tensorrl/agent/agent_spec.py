import tensorflow as tf
import collections
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import nest
import six



class AgentSpec(
    collections.namedtuple('AgentSpec', tf.estimator.EstimatorSpec._fields + (
        'step_hooks', 'episode_metric_ops',
    ))):
    
    def __new__(cls,
        # estimator
        mode,
        predictions=None,
        loss=None,
        train_op=None,
        eval_metric_ops=None,
        export_outputs=None,
        training_chief_hooks=None,
        training_hooks=None,
        scaffold=None,
        evaluation_hooks=None,
        prediction_hooks=None,
        # agent
        step_hooks=None,
        episode_metric_ops=None,
        ):

        estimator_spec = tf.estimator.EstimatorSpec.__new__(
            tf.estimator.EstimatorSpec,
            mode,
            predictions,
            loss,
            train_op,
            eval_metric_ops,
            export_outputs,
            training_chief_hooks,
            training_hooks,
            scaffold,
            evaluation_hooks,
            prediction_hooks,
        )

        # Validate hooks.
        step_hooks = tuple(step_hooks or [])


        for hook in step_hooks:
            if not isinstance(hook, session_run_hook.SessionRunHook):
                raise TypeError(
                    'All hooks must be SessionRunHook instances, given: {}'.format(hook)
                )

        # Validate episode_metric_ops.
        if episode_metric_ops is None:
            episode_metric_ops = {}
        else:
            if not isinstance(episode_metric_ops, dict):
                raise TypeError(
                    'episode_metric_ops must be a dict, given: {}'.format(episode_metric_ops))
            for key, metric_value_and_update in six.iteritems(episode_metric_ops):
                if (not isinstance(metric_value_and_update, tuple) or
                    len(metric_value_and_update) != 2):
                    raise TypeError(
                        'Values of episode_metric_ops must be (metric_value, update_op) '
                        'tuples, given: {} for key: {}'.format(
                            metric_value_and_update, key))
                metric_value, metric_update = metric_value_and_update
                for metric_value_member in nest.flatten(metric_value):
                    # Allow (possibly nested) tuples for metric values, but require that
                    # each of them be Tensors or Operations.
                    _check_is_tensor_or_operation(metric_value_member,
                                                'episode_metric_ops[{}]'.format(key))
                _check_is_tensor_or_operation(metric_update,
                                            'episode_metric_ops[{}]'.format(key))

        
        args = tuple(estimator_spec) + (step_hooks, prediction_hooks)
        
        return super(AgentSpec, cls).__new__(cls, *args)



def _check_is_tensor_or_operation(x, name):
    if not (isinstance(x, tf.Operation) or isinstance(x, tf.Tensor)):
        raise TypeError('{} must be Operation or Tensor, given: {}'.format(name, x))


def _check_is_tensor(x, tensor_name):
    """Returns `x` if it is a `Tensor`, raises TypeError otherwise."""
    if not isinstance(x, tf.Tensor):
        raise TypeError('{} must be Tensor, given: {}'.format(tensor_name, x))
    return x


if __name__ == '__main__':
    
    spec = AgentSpec(tf.estimator.ModeKeys.PREDICT, predictions = tf.placeholder(tf.float32))
    

    print(spec)