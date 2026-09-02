from rest_framework import serializers

from .models import UserSecret, validate_secret_name


class UserSecretSerializer(serializers.ModelSerializer):
    is_set = serializers.BooleanField(read_only=True)
    value = serializers.CharField(write_only=True, required=False, allow_blank=False)

    class Meta:
        model = UserSecret
        fields = [
            "id",
            "name",
            "description",
            "is_set",
            "value",
            "created_at",
            "updated_at",
            "last_used_at",
        ]
        read_only_fields = [
            "id",
            "is_set",
            "created_at",
            "updated_at",
            "last_used_at",
        ]

    def validate_name(self, value):
        validate_secret_name(value)
        return value
