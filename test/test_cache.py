# pylint: disable=no-self-use
from os import listdir
import sets


class TestDiskCache:

    def test_only_called_once(self, tmpdir):
        called = 0
        @sets.disk_cache('foo', str(tmpdir))
        def pipeline():
            nonlocal called
            called += 1
            return 'Foo'
        assert called == 0
        assert pipeline() == 'Foo'
        assert called == 1
        assert pipeline() == 'Foo'
        assert called == 1

    def test_new_argument(self, tmpdir):
        @sets.disk_cache('foo', str(tmpdir))
        def pipeline(argument):
            return 2 * argument
        assert pipeline(1) == 2
        assert pipeline(2) == 4

    def test_method_argument(self, tmpdir):
        called = 0
        @sets.disk_cache('foo', str(tmpdir), method=True)
        def pipeline(self):
            # pylint: disable=unused-argument
            nonlocal called
            called += 1
            return 'Foo'
        assert called == 0
        assert pipeline(None) == 'Foo'
        assert called == 1
        assert pipeline(None) == 'Foo'
        assert called == 1

    def test_kwarg_hash(self, tmpdir):
        @sets.disk_cache('foo', str(tmpdir), method=True)
        def dummy(self, arg, kwarg=False):
            pass
        dummy(None, "foo")
        dummy(None, "foo", kwarg=True)
        assert(2 == len(listdir(str(tmpdir))))
