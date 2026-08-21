from rest_framework import serializers


class MessageSerializer(serializers.Serializer):
    """One prior conversation turn."""

    role = serializers.ChoiceField(choices=["user", "assistant", "system"])
    content = serializers.CharField(allow_blank=True)


class ChatRequestSerializer(serializers.Serializer):
    """Input to POST /api/chat/."""

    question = serializers.CharField(min_length=1, max_length=2000, trim_whitespace=True)
    history = MessageSerializer(many=True, required=False, default=list)
    top_k = serializers.IntegerField(required=False, min_value=1, max_value=20)


class SourceSerializer(serializers.Serializer):
    """A document that backed the answer."""

    filename = serializers.CharField()
    heading = serializers.CharField(allow_blank=True)
    chunk_index = serializers.IntegerField(allow_null=True)
    score = serializers.FloatField(allow_null=True)


class ChatResponseSerializer(serializers.Serializer):
    """Output of POST /api/chat/."""

    answer = serializers.CharField()
    sources = SourceSerializer(many=True)
